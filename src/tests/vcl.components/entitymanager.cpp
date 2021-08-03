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

// VCL configuration
#include <vcl/config/global.h>

// Include the relevant parts from the library
#include <vcl/components/entitymanager.h>

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

struct NameComponent
{
	NameComponent() = default;
	NameComponent(const std::string& name)
	: Name(name) {}
	NameComponent(const char* name)
	: Name(name) {}

	std::string Name;
};

struct SecondaryNameComponent
{
	SecondaryNameComponent() = default;
	SecondaryNameComponent(const std::string& name)
	: Name(name) {}
	SecondaryNameComponent(const char* name)
	: Name(name) {}

	std::string Name;
};

namespace Vcl { namespace Components {
	template<>
	struct ComponentTraits<SecondaryNameComponent>
	{
		static const bool IsUnique{ false };
	};
}}

TEST(EntityManagerTest, CreateDestroyEntites)
{
	using namespace Vcl::Components;

	EntityManager em;

	// Create four entities
	auto e0 = em.create();
	auto e1 = em.create();
	auto e2 = em.create();
	auto e3 = em.create();

	// Check that all four entites were allocated
	EXPECT_EQ(4, em.size());
	EXPECT_EQ(4, em.capacity());

	// Destroy two of the entities
	em.destroy(e1);
	em.destroy(e2);

	EXPECT_EQ(2, em.size());
	EXPECT_EQ(4, em.capacity());
}

TEST(EntityManagerTest, CreateDestroyUniqueComponents)
{
	using namespace Vcl::Components;

	EntityManager em;
	auto e0 = em.create();
	auto e1 = em.create();
	auto e2 = em.create();
	auto e3 = em.create();

	em.registerComponent<NameComponent>();

	auto c0 = em.create<NameComponent>(e0, "E0");
	auto c2 = em.create<NameComponent>(e2, "E2");
	auto c3 = em.create<NameComponent>(e3, "E3");

	EXPECT_TRUE(em.has<NameComponent>(e0));
	EXPECT_FALSE(em.has<NameComponent>(e1));
	EXPECT_TRUE(em.has<NameComponent>(e2));
	EXPECT_TRUE(em.has<NameComponent>(e3));

	EXPECT_EQ("E0", c0->Name);
	EXPECT_EQ("E2", c2->Name);
	EXPECT_EQ("E3", c3->Name);
}

TEST(EntityManagerTest, CreateDestroyMultiComponents)
{
	using namespace Vcl::Components;

	EntityManager em;
	auto e0 = em.create();
	auto e1 = em.create();
	auto e2 = em.create();
	auto e3 = em.create();

	em.registerComponent<SecondaryNameComponent>([](const SecondaryNameComponent& c, const std::string& s) {
		return c.Name == s;
	});

	auto c0 = em.create<SecondaryNameComponent>(e0, "E0");
	auto c2_1 = em.create<SecondaryNameComponent>(e2, "E2_1");
	auto c2_2 = em.create<SecondaryNameComponent>(e2, "E2_2");

	EXPECT_TRUE(em.has<SecondaryNameComponent>(e0));
	EXPECT_FALSE(em.has<SecondaryNameComponent>(e1));
	EXPECT_TRUE(em.has<SecondaryNameComponent>(e2));
	EXPECT_FALSE(em.has<SecondaryNameComponent>(e3));

	EXPECT_EQ("E0", c0->Name);
	EXPECT_EQ("E2_1", c2_1->Name);
	EXPECT_EQ("E2_2", c2_2->Name);
}
