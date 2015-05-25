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

// Google test
#include <gtest/gtest.h>

struct NameComponent
{
	NameComponent() = default;
	NameComponent(const std::string& name) : Name(name) {}

	std::string Name;
};

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

	auto c0 = em.create<NameComponent>(e0, "E0");
	auto c2 = em.create<NameComponent>(e2, "E2");
	auto c3 = em.create<NameComponent>(e3, "E3");

	EXPECT_TRUE (em.has<NameComponent>(e0));
	EXPECT_FALSE(em.has<NameComponent>(e1));
	EXPECT_TRUE (em.has<NameComponent>(e2));
	EXPECT_TRUE (em.has<NameComponent>(e3));

	EXPECT_EQ("E0", c0->Name);
	EXPECT_EQ("E2", c2->Name);
	EXPECT_EQ("E3", c3->Name);
}
