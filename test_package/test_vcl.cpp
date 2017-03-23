#include <vcl/core/handle.h>
#include <iostream>

int main(int argc, char** argv)
{
	std::cout << Vcl::createResourceHandleTag(nullptr) << std::endl;
	return 0;
}
