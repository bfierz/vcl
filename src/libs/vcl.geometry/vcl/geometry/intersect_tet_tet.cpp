/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <vcl/geometry/intersect.h>

namespace Vcl { namespace Geometry
{// ----------- 3D algebraic operators -------------


#define DOT(a,b) (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])

#define VECT(res,a, b){ \
	res[0] = a[1]*b[2]-b[1]*a[2];\
	res[1] = -a[0]*b[2]+b[0]*a[2];\
	res[2] = a[0]*b[1]-b[0]*a[1];\
}

#define SUB(res,a,b) {\
	res[0] = a[0]-b[0];\
	res[1] = a[1]-b[1];\
	res[2] = a[2]-b[2];\
}

#define SUB_DOT(a,b,c) (\
	 (a[0]-b[0])*c[0]+\
	 (a[1]-b[1])*c[1]+\
	 (a[2]-b[2])*c[2]\
)




	typedef double point[3];
	static point *V1, *V2;			        // vertices coordinates


	static double e_v1[6][3], e_v2[6][3];            // vectors edge-oriented


	static int masks[4];			        // for each face of the first tetrahedron

											// stores the halfspace each vertex of the

											// second tetrahedron belongs to


	static double P_V1[4][3], P_V2[4][3];           // differences between the vertices of the second (first) 

													//  tetrahedron

													// and the vertex 0  of the first(second) tetrahedron

	static double  Coord_1[4][4], Coord_2[4][4];     // vertices coordinates in the affine space


	static double n[3];			        // variable to store the normals



										// FaceA ----------------------------------------------------

	inline static bool FaceA_1(double * Coord, int & maskEdges)
	{

		maskEdges = 000;

		if ((Coord[0] = DOT(P_V1[0], n)) > 0) maskEdges = 001;
		if ((Coord[1] = DOT(P_V1[1], n)) > 0) maskEdges |= 002;
		if ((Coord[2] = DOT(P_V1[2], n)) > 0) maskEdges |= 004;
		if ((Coord[3] = DOT(P_V1[3], n)) > 0) maskEdges |= 010;


		return (maskEdges == 017);	// if true it means that all of the vertices are out the halfspace

									// defined by this face

	}
	// it is the same as FaceA_1, only the values V2[0]-v_ref are used only for the fourth face

	// hence they do not need to be stored

	inline static bool FaceA_2(double * Coord, int & maskEdges)
	{
		maskEdges = 000;
		double * v_ref = V1[1];

		if ((Coord[0] = SUB_DOT(V2[0], v_ref, n)) > 0) maskEdges = 001;
		if ((Coord[1] = SUB_DOT(V2[1], v_ref, n)) > 0) maskEdges |= 002;
		if ((Coord[2] = SUB_DOT(V2[2], v_ref, n)) > 0) maskEdges |= 004;
		if ((Coord[3] = SUB_DOT(V2[3], v_ref, n)) > 0) maskEdges |= 010;

		return (maskEdges == 017);
	}



	// FaceB --------------------------------------------------------------

	inline static bool FaceB_1()
	{

		return  ((DOT(P_V2[0], n)>0) &&
			(DOT(P_V2[1], n)>0) &&
			(DOT(P_V2[2], n)>0) &&
			(DOT(P_V2[3], n)>0));
	}

	inline static bool FaceB_2()
	{
		double * v_ref = V2[1];
		return	((SUB_DOT(V1[0], v_ref, n) > 0) &&
			(SUB_DOT(V1[1], v_ref, n) > 0) &&
			(SUB_DOT(V1[2], v_ref, n) > 0) &&
			(SUB_DOT(V1[3], v_ref, n) > 0));
	}


	// EdgeA -------------------------------------------------------

	inline static bool EdgeA(const int & f0, const int & f1)
	{

		double * coord_f0 = &Coord_1[f0][0];
		double * coord_f1 = &Coord_1[f1][0];

		int  maskf0 = masks[f0];
		int  maskf1 = masks[f1];

		if ((maskf0 | maskf1) != 017) // if there is a vertex of b 

			return false;	      // included in (-,-) return false


		maskf0 &= (maskf0 ^ maskf1);  // exclude the vertices in (+,+)

		maskf1 &= (maskf0 ^ maskf1);

		// edge 0: 0--1 

		if (((maskf0 & 001) &&		// the vertex 0 of b is in (-,+) 

			(maskf1 & 002)) &&		// the vertex 1 of b is in (+,-)

			(((coord_f0[1] * coord_f1[0]) -
			(coord_f0[0] * coord_f1[1]))  >0))
			// the edge of b (0,1) intersect (-,-) (see the paper)

			return false;

		if (((maskf0 & 002) && (maskf1 & 001)) && (((coord_f0[1] * coord_f1[0]) - (coord_f0[0] * coord_f1[1]))  < 0))
			return false;

		// edge 1: 0--2 

		if (((maskf0 & 001) && (maskf1 & 004)) && (((coord_f0[2] * coord_f1[0]) - (coord_f0[0] * coord_f1[2]))  > 0))
			return false;

		if (((maskf0 & 004) && (maskf1 & 001)) && (((coord_f0[2] * coord_f1[0]) - (coord_f0[0] * coord_f1[2]))  < 0))
			return false;

		// edge 2: 0--3 

		if (((maskf0 & 001) && (maskf1 & 010)) && (((coord_f0[3] * coord_f1[0]) - (coord_f0[0] * coord_f1[3]))  > 0))
			return false;

		if (((maskf0 & 010) && (maskf1 & 001)) && (((coord_f0[3] * coord_f1[0]) - (coord_f0[0] * coord_f1[3]))  < 0))
			return false;

		// edge 3: 1--2 

		if (((maskf0 & 002) && (maskf1 & 004)) && (((coord_f0[2] * coord_f1[1]) - (coord_f0[1] * coord_f1[2]))  > 0))
			return false;

		if (((maskf0 & 004) && (maskf1 & 002)) && (((coord_f0[2] * coord_f1[1]) - (coord_f0[1] * coord_f1[2]))  < 0))
			return false;


		// edge 4: 1--3 

		if (((maskf0 & 002) && (maskf1 & 010)) && (((coord_f0[3] * coord_f1[1]) - (coord_f0[1] * coord_f1[3]))  > 0))
			return false;

		if (((maskf0 & 010) && (maskf1 & 002)) && (((coord_f0[3] * coord_f1[1]) - (coord_f0[1] * coord_f1[3]))  < 0))
			return false;

		// edge 5: 2--3 

		if (((maskf0 & 004) && (maskf1 & 010)) && (((coord_f0[3] * coord_f1[2]) - (coord_f0[2] * coord_f1[3])) > 0))
			return false;

		if (((maskf0 & 010) && (maskf1 & 004)) && (((coord_f0[3] * coord_f1[2]) - (coord_f0[2] * coord_f1[3])) < 0))
			return false;

		return true;	// there exists a separting plane supported by the edge shared by f0 and f1

	}

	// main function

	bool tet_a_tet(double V_1[4][3], double V_2[4][3])
	{

		V1 = V_1;
		V2 = V_2;

		SUB(P_V1[0], V2[0], V1[0]);
		SUB(P_V1[1], V2[1], V1[0]);
		SUB(P_V1[2], V2[2], V1[0]);
		SUB(P_V1[3], V2[3], V1[0]);


		SUB(e_v1[0], V1[1], V1[0]);
		SUB(e_v1[1], V1[2], V1[0]);

		VECT(n, e_v1[0], e_v1[1]);		// find the normal to  face 0


		if (FaceA_1(&Coord_1[0][0], masks[0]))	return false;


		SUB(e_v1[2], V1[3], V1[0]);
		VECT(n, e_v1[2], e_v1[0]);

		if (FaceA_1(&Coord_1[1][0], masks[1])) 	return false;

		if (EdgeA(0, 1)) return false;


		VECT(n, e_v1[1], e_v1[2]);

		if (FaceA_1(&Coord_1[2][0], masks[2])) 	return false;

		if (EdgeA(0, 2)) return false;
		if (EdgeA(1, 2)) return false;

		SUB(e_v1[4], V1[3], V1[1]);
		SUB(e_v1[3], V1[2], V1[1]);

		VECT(n, e_v1[4], e_v1[3]);

		if (FaceA_2(&Coord_1[3][0], masks[3]))  return false;

		if (EdgeA(0, 3)) return false;
		if (EdgeA(1, 3)) return false;
		if (EdgeA(2, 3)) return false;

		if ((masks[0] | masks[1] | masks[2] | masks[3]) != 017) return true;


		// from now on, if there is a separating plane it is parallel to a face of b

		SUB(P_V2[0], V1[0], V2[0]);
		SUB(P_V2[1], V1[1], V2[0]);
		SUB(P_V2[2], V1[2], V2[0]);
		SUB(P_V2[3], V1[3], V2[0]);


		SUB(e_v2[0], V2[1], V2[0]);
		SUB(e_v2[1], V2[2], V2[0]);

		VECT(n, e_v2[0], e_v2[1]);
		if (FaceB_1()) return false;

		SUB(e_v2[2], V2[3], V2[0]);

		VECT(n, e_v2[2], e_v2[0]);

		if (FaceB_1()) return false;

		VECT(n, e_v2[1], e_v2[2]);

		if (FaceB_1()) return false;

		SUB(e_v2[4], V2[3], V2[1]);
		SUB(e_v2[3], V2[2], V2[1]);

		VECT(n, e_v2[4], e_v2[3]);

		if (FaceB_2()) return false;

		return true;
	}

#undef DOT
#undef SUB
#undef SUB_DOT
#undef VECT
	bool intersects(const Tetrahedron<float, 3>& t0, const Tetrahedron<float, 3>& t1)
	{
		double V_1[4][3];
		double V_2[4][3];
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 3; j++)
			{
				V_1[i][j] = t0[i][j];
				V_2[i][j] = t1[i][j];
			}

		return tet_a_tet(V_1, V_2);
	}
}}
