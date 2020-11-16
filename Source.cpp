/*
 * stijn: I wrote this small test case while debugging the "invisible collision" problem in UT469.
 * When dodging into or sliding over a slope at high FPS, you often get stuck for no apparent
 * reason. There are two root causes for this problem:
 *
 * + The code that adjusts your movement vector when a collision is imminent is supposed to
 * calculate a vector that is parallel to the plane, but due to 32-bit float imprecision it
 * might still intersect the plane.
 *
 * + The code that detects collisions compares the distance between the current location and the
 * collision plane (D0), with the current location + adjusted movement vector and the plane (D1).
 * If we are far away from the world space origin, then D1 might be way off. 
 */
#include <emmintrin.h>
#include <sstream>
#include <stdio.h>
#include <string>

#define DOUBLE_PRECISION 1

#if DOUBLE_PRECISION
typedef double FLOAT;
#else
typedef float FLOAT;
#endif
typedef unsigned int DWORD;
typedef int UBOOL;

class FVector
{
public:
	FLOAT X, Y, Z;

	FVector(FLOAT InX, FLOAT InY, FLOAT InZ)
		: X(InX)
		, Y(InY)
		, Z(InZ)
	{}

	FVector(DWORD InXBits, DWORD InYBits, DWORD InZBits)
		: X(*reinterpret_cast<float*>(&InXBits))
		, Y(*reinterpret_cast<float*>(&InYBits))
		, Z(*reinterpret_cast<float*>(&InZBits))
	{}

	inline FVector operator-(const FVector& V) const
	{
		return FVector(X - V.X, Y - V.Y, Z - V.Z);
	}

	inline FVector operator*(FLOAT Scale) const
	{
		return FVector(X * Scale, Y * Scale, Z * Scale);
	}

	inline FLOAT operator|(const FVector& V) const
	{
		return X * V.X + Y * V.Y + Z * V.Z;
	}

	inline FLOAT Size() const
	{
		return sqrtf(X * X + Y * Y + Z * Z);
	}

	inline FVector operator/(FLOAT Scale) const
	{
		FLOAT RScale = 1.f / Scale;
		return FVector(X * RScale, Y * RScale, Z * RScale);
	}

	friend FVector operator*(FLOAT Scale, const FVector& V);

	inline FVector operator+(const FVector& V) const
	{
		return FVector(X + V.X, Y + V.Y, Z + V.Z);
	}

	std::string String()
	{
		std::stringstream ss;
		ss.precision(10);
		ss << "(X=" << X << ", Y=" << Y << ", Z=" << Z << ")";
		return ss.str();
	}
};

inline FVector operator*(FLOAT Scale, const FVector& V)
{
	return FVector(V.X * Scale, V.Y * Scale, V.Z * Scale);
}

class FPlane
{
public:
	FLOAT X, Y, Z, W;

	FPlane(FLOAT InX, FLOAT InY, FLOAT InZ, FLOAT InW)
		: X(InX)
		, Y(InY)
		, Z(InZ)
		, W(InW)
	{}

	FPlane(DWORD InXBits, DWORD InYBits, DWORD InZBits, DWORD InWBits)
		: X(*reinterpret_cast<float*>(&InXBits))
		, Y(*reinterpret_cast<float*>(&InYBits))
		, Z(*reinterpret_cast<float*>(&InZBits))
		, W(*reinterpret_cast<float*>(&InWBits))
	{}

	FVector Normal()
	{
		return FVector(X, Y, Z);
	}

#if !DOUBLE_PRECISION
	FLOAT PlaneDotSSE(const FVector& P) const;
#endif
	FLOAT PlaneDot(const FVector& P) const;

	std::string String()
	{
		std::stringstream ss;
		ss.precision(10);
		ss << "(X=" << X << ", Y=" << Y << ", Z=" << Z << ", W=" << W << ")";
		return ss.str();
	}
};

#if !DOUBLE_PRECISION
inline __m128 _mm(const FPlane& P)
{
	return _mm_loadu_ps(&P.X);
}

// Load FVector to XMM safely (we can specify whether we want to zero the unused float)
template<UBOOL bZeroW = 1>
inline __m128 _mm(const FVector& V)
{
#if VECTOR_ALIGNMENT == 16
	__m128 mm = _mm_loadu_ps(&V.X);  //X,Y,Z,?
	if (bZeroW)
		mm = _mm_and_ps(mm, _mm_castsi128_ps(MM_3D_MASK)); //X,Y,Z,0
#else
	__m128 mm = _mm_setzero_ps();    //0,0,0,0 (some compilers already do this prior to _mm_load_ss)
	mm = _mm_load_ss(&V.Z);        //Z,0,0,0
	mm = _mm_movelh_ps(mm, mm);     //Z,0,Z,0
	mm = _mm_loadl_pi(mm, (const __m64*) & V.X); //X,Y,Z,0 (we load the low 8 bytes onto mm)
#endif
	return mm;
}

inline __m128 _mm_coords_sum_ps(__m128 x)
{
	__m128 w = _mm_shuffle_ps(x, x, 0b10110001); //x,y,z,w >> y,x,w,z
//	__m128 w = _mm_pshufd_ps( v, 0b10110001); /*SSE2 version*/
	x = _mm_add_ps(x, w); // x+y,-,z+w,-
	w = _mm_movehl_ps(w, x); // >> z+w,-,-,-
	w = _mm_add_ss(w, x); // x+y+z+w,-,-,-
	return w;
}

inline FLOAT FPlane::PlaneDotSSE(const FVector& P) const
{
	// Used to fill the W coordinate with -1
	constexpr __m128 MM_PLANEDOT_W{ 0.f, 0.f, 0.f, -1.f };

	FLOAT Result;
	__m128 mm_x_y_z_w = _mm_or_ps(_mm(P), MM_PLANEDOT_W); //PX,PY,PZ,-1
	mm_x_y_z_w = _mm_mul_ps(mm_x_y_z_w, _mm(*this)); //X*PX, Y*PY, Z*PZ, -W   (x,y,z,w)
	_mm_store_ss(&Result, _mm_coords_sum_ps(mm_x_y_z_w));
	return Result;
}
#endif

inline FLOAT FPlane::PlaneDot(const FVector& P) const
{
	return X * P.X + Y * P.Y + Z * P.Z - W;
}

inline FLOAT FBoxPushOut(FVector Normal, FVector Size)
{
	return abs(Normal.X * Size.X) + abs(Normal.Y * Size.Y) + abs(Normal.Z * Size.Z);
}

int main(int argc, char** argv)
{
	FLOAT T0 = -1.f;
	FLOAT T1 = 1.f;
	FLOAT HitTime = 0.f;

	// UT: Start (X=-20777.466797,Y=-6414.467285,Z=-2672.729248) - (0xc6a252ef,0xc5c873bd,0xc5270bab)
	// Unit Test: Start = (X=-20777.4668, Y=-6414.467285, Z=-2672.729248)
	FVector Start(0xc6a252ef, 0xc5c873bd, 0xc5270bab);
	// Log: Trying Delta (X=-0.135223,Y=0.000000,Z=-0.162531) - (0xbe0a77c5,0x00000000,0xbe266e99)
	// Unit Test: Delta = (X=-0.1352225095, Y=0, Z=-0.1625312716)
	FVector Delta(0xbe0a77c5, 0x00000000, 0xbe266e99);
	// Log: Hit Collision Plane: (X=-0.761811,Y=-0.000000,Z=0.647799,W=14058.916016)- (0xbf430613,0xb40451c4,0x3f25d625,0x465babaa)
	// Unit Test: CollisionPlane = (X=-0.7618114352, Y=-1.232320415e-07, Z=0.6477988362, W=14058.91602)
	FPlane CollisionPlane(0xbf430613, 0xb40451c4, 0x3f25d625, 0x465babaa);

	// Log: Deflected Delta is (X=-0.136955,Y=-0.000000,Z=-0.161058) - (0xbe0c3dcd,0xaf9a0668,0xbe24ec84) (0.211415 UU)
	// Unit Test: DeflectedDelta (UT) = (X=-0.1369545013, Y=-2.801698873e-10, Z=-0.1610584855)
	FVector DeflectedDelta(0xbe0c3dcd, 0xaf9a0668, 0xbe24ec84);
	// FVector DeflectedDelta = (Delta - CollisionPlane.Normal * (Delta | CollisionPlane.Normal)) * (1.f - HitTime);
	FVector DeflectedDeltaRecalc = (Delta - CollisionPlane.Normal() * (Delta | CollisionPlane.Normal())) * (1.f - HitTime);

	// Log: TestDelta is (X=-1.432552,Y=-0.000000,Z=-1.684681) - (0xbfb75de0,0xb149638d,0xbfd7a3a4) (2.211415 UU)
	// Unit Test: TestDelta (UT) = (X=-1.432552338, Y=-2.930593768e-09, Z=-1.684681416)
	FVector TestDelta(0xbfb75de0, 0xb149638d, 0xbfd7a3a4);
	//
	FLOAT DeflectedDeltaSize  = DeflectedDeltaRecalc.Size();
	FVector DeflectedDeltaDir = DeflectedDeltaRecalc / DeflectedDeltaSize;
	FVector TestDeltaRecalc   = DeflectedDeltaRecalc + 2.f * DeflectedDeltaDir;

	// Log: End is (X=-20778.898438,Y=-6414.467285,Z=-2674.413818) - (0xc6a255cc,0xc5c873bd,0xc527269f)
	FVector End(0xc6a255cc, 0xc5c873bd, 0xc527269f);
	//
	FVector EndRecalc(Start.X + TestDeltaRecalc.X, Start.Y + TestDeltaRecalc.Y, Start.Z + TestDeltaRecalc.Z);
	
	// Log: D0 is 38.206055 (0x4218d300) D1 is 38.205078 (0x4218d200)
	DWORD D0Raw = 0x4218d300;
	DWORD D1Raw = 0x4218d200;
	FLOAT D0 = *reinterpret_cast<float*>(&D0Raw);
	FLOAT D1 = *reinterpret_cast<float*>(&D1Raw);
	
#if !DOUBLE_PRECISION
	FLOAT D0RecalcSSE = CollisionPlane.PlaneDotSSE(Start);
	FLOAT D1RecalcSSE = CollisionPlane.PlaneDotSSE(EndRecalc);
#endif
	FLOAT D0Recalc = CollisionPlane.PlaneDot(Start);
	FLOAT D1Recalc = CollisionPlane.PlaneDot(EndRecalc);

	// Calculate time until hit
	FVector Extent(17.f, 17.f, 39.f);
	FLOAT PushOut = FBoxPushOut(CollisionPlane.Normal(), Extent);

	FLOAT AdjD0 = D0Recalc - PushOut;
	if (D0Recalc > D1Recalc && AdjD0 >= -PushOut && AdjD0 < 0)
		AdjD0 = 0.f;

	FLOAT T = (AdjD0) / (D0Recalc - D1Recalc);
	if (T > HitTime)
		HitTime = T;

	// Attenuate Movement
	FVector FinalDelta = DeflectedDeltaRecalc;
	if (HitTime < 1.f)
	{
		FLOAT FinalDeltaSize = (DeflectedDeltaSize + 2.f) * HitTime;
		if (FinalDeltaSize <= 2.f)
		{
			HitTime = 0;
			FinalDelta = FVector(0.f, 0.f, 0.f);
		}
		else
		{
			FinalDelta = TestDeltaRecalc * HitTime - 2.f * DeflectedDeltaDir;
			HitTime    = (FinalDeltaSize - 2.f) / DeflectedDeltaSize;
		}
	}

	printf("########################################################################################################################\n");
	printf("#                                    Unreal Tournament v469 Movement Simulation                                        #\n");
	printf("########################################################################################################################\n");
	printf("> Recalculation Precision: %s\n",
#if DOUBLE_PRECISION
		"Double"
#else
		"Single"
#endif
		);
	printf("------------------------------------------------------------------------------------------------------------------------\n");
	printf("> Start = %s\n", Start.String().c_str());
	printf("> Delta = %s\n", Delta.String().c_str());
	printf("> CollisionPlane = %s\n", CollisionPlane.String().c_str());
	printf("------------------------------------------------------------------------------------------------------------------------\n");
	printf("> DeflectedDelta (UT) = %s\n", DeflectedDelta.String().c_str());
	printf("> DeflectedDelta (Recalculated) = %s - Rounding Error = %f\n", DeflectedDeltaRecalc.String().c_str(), (DeflectedDeltaRecalc-DeflectedDelta).Size());
	printf("------------------------------------------------------------------------------------------------------------------------\n");
	printf("> TestDelta (UT) = %s\n", TestDelta.String().c_str());
	printf("> TestDelta (Recalculated) = %s - Rounding Error = %f\n", TestDeltaRecalc.String().c_str(), (TestDeltaRecalc-TestDelta).Size());
	printf("------------------------------------------------------------------------------------------------------------------------\n");
	printf("> End (UT) = %s\n", End.String().c_str());
	printf("> End (Recalculated) = %s - Rounding Error = %f\n", EndRecalc.String().c_str(), (EndRecalc - End).Size());
	printf("------------------------------------------------------------------------------------------------------------------------\n");
	printf("> D0 (UT) = %f\n", D0);
	printf("> D1 (UT) = %f\n", D1);
	printf("> Collision? (UT) = %s\n", ((D0 - D1) > +0.00001f) ? "YES" : "NO");
	printf("------------------------------------------------------------------------------------------------------------------------\n");

	printf("> D0 (Recalculated on FPU) = %f\n", D0Recalc);
	printf("> D1 (Recalculated on FPU) = %f\n", D1Recalc);
	printf("> Collision? (Recalcalculated on FPU) = %s\n", ((D0Recalc - D1Recalc) > +0.00001f) ? "YES" : "NO");
	printf("------------------------------------------------------------------------------------------------------------------------\n");

#if !DOUBLE_PRECISION
	printf("> D0 (Recalculated with SSE) = %f\n", D0RecalcSSE);
	printf("> D1 (Recalculated with SSE) = %f\n", D1RecalcSSE);
	printf("> Collision? (Recalculated with SSE) = %s\n", ((D0RecalcSSE - D1RecalcSSE) > +0.00001f) ? "YES" : "NO");
	printf("------------------------------------------------------------------------------------------------------------------------\n");
#endif

	printf("> AdjD0 = %f\n", AdjD0);
	printf("> Final HitTime = %f\n", HitTime);
	printf("> Final Delta = %s\n", FinalDelta.String().c_str());
	printf("------------------------------------------------------------------------------------------------------------------------\n");
	
	return 0;
}