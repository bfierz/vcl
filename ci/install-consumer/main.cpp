// Touches one symbol of each library that is built without graphics dependencies, so that
// the consumer fails to compile or link if a vcl:: target is incomplete.
#include <vcl/components/entitymanager.h>
#include <vcl/core/contract.h>
#include <vcl/geometry/distancePoint3Triangle3.h>
#include <vcl/math/solver/conjugategradients.h>

#include <cstdio>

int main()
{
	Vcl::Components::EntityManager entities;
	auto entity = entities.create();
	entities.destroy(entity);

	const Eigen::Vector3f a{ 0, 0, 0 }, b{ 1, 0, 0 }, c{ 0, 1, 0 };
	const Vcl::Geometry::Triangle<float, 3> triangle{ a, b, c };
	const float d = Vcl::Geometry::distance(triangle, Eigen::Vector3f{ 0.25f, 0.25f, 1 });

	Vcl::Mathematics::Solver::ConjugateGradients cg;
	cg.setPrecision(1e-6);

	std::printf("vcl consumer: %f distance, %d iterations\n", d, cg.nrIterations());
	return 0;
}
