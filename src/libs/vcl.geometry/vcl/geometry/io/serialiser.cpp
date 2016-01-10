#include <vcl/geometry/io/serialiser.h>

// C++ standard library
#include <fstream>
#include <sstream>

namespace Vcl { namespace Geometry { namespace IO
{	
	std::string Serialiser::readToken(std::istream& fin) const
	{
		std::stringstream result;
		std::string::value_type c;
		fin.read(&c, sizeof(std::string::value_type));
		bool quoted = false;

		// omit white spaces
		while (!fin.eof() && c != -1 && c < 33) 
			fin.read(&c, sizeof(std::string::value_type));

		// is there at all a non-white space character
		if (fin.eof())
			return std::string();

		// is the token a string
		if (c == '\"')
			quoted = true;

		// read the token
		do
		{
			result << c;
			fin.read(&c, sizeof(std::string::value_type));
			if (c == '\"')
				quoted = false;
		} while (!fin.eof() && c != -1 && (quoted || c >= 33));

		return result.str();
	}
	
	void Serialiser::skipLine(std::ifstream& fin) const
	{
		std::string buffer;
		std::getline(fin, buffer);
	}
	
	void Serialiser::readLine(std::ifstream& fin, std::stringstream& output) const
	{
		std::string buffer;
		std::getline(fin, buffer);

		output.clear();
		output.str(buffer);
	}
	
	float Serialiser::readFloat(std::istream& fin) const
	{
		std::string token = std::move(readToken(fin));
		if (token.length() == 0)
			return float(0);
		else
			return (float) atof(token.c_str());
	}

	int Serialiser::readInteger(std::istream& fin) const
	{
		std::string token = std::move(readToken(fin));
		return convertTokenToInteger(token);
	}

	unsigned int Serialiser::readUnsignedInteger(std::istream& fin) const
	{
		std::string token = std::move(readToken(fin));
		if (token.length() == 0)
			return 0;
		else
			return static_cast<unsigned int>(atol(token.c_str()));
	}
	
	std::string Serialiser::readString(std::ifstream& fin) const
	{
		std::string token = std::move(readToken(fin));
		if (token.length() == 0)
			return token;
		else
			return token.substr(1, token.length() - 2);
	}

	int Serialiser::convertTokenToInteger(const std::string& token) const
	{
		if (token.length() == 0)
			return 0;
		else
			return atoi(token.c_str());
	}
}}}
