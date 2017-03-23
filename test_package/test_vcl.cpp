#include <vcl/core/handle.h>
#include <iostream>

int main(int argc, char** argv)
{
	auto handle_tag = Vcl::createResourceHandleTag(nullptr);
	std::cout << "Success" << std::endl;
	return 0;
}
