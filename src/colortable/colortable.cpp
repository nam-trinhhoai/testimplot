#include "colortable.h"

#include <sstream>
#include <fstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
ColorTable::ColorTable() {

}

ColorTable::ColorTable(const ColorTable &par) {
	this->m_colors = par.m_colors;
	this->m_name = par.m_name;
}

ColorTable& ColorTable::operator=(const ColorTable &par) {
	if (this != &par) {
		this->m_colors = par.m_colors;
		this->m_name = par.m_name;
	}
	return *this;
}

bool ColorTable::operator==(const ColorTable &par) {
	if( this == &par )
		return true;

	return (this->m_colors == par.m_colors) &&
			(this->m_name == par.m_name);
}

bool ColorTable::readFile(const std::string &path) {

	std::ifstream infile(path);
	if (!infile.is_open())
		return false;

	//Read header #ColorTable Azimuth-1
	std::string line;
	if (!std::getline(infile, line))
		return false;

	if (line.find("ColorTable") != std::string::npos) {
		std::istringstream iss(line);
		std::string e;
		if (!(iss >> e >> m_name))
			return false;
	} else
		return false;

	while (line.find("#") != std::string::npos) {
		if (!std::getline(infile, line))
			return false;
	}

	std::map<int, std::array<int, 4>> colors;
	//Read foreground
	while (true) {
		std::string colorName = line.substr(0, line.find("="));
		std::string color = line.substr(line.find("=") + 1);

		std::istringstream iss(color);
		std::array<int, 4> col;
		if (!(iss >> col[0] >> col[1] >> col[2]))
			break;

		col[3] = 255;
		if (colorName != "foreground" && colorName != "background") {
			int pos = atoi(colorName.substr(3).c_str());
			colors[pos] = col;
		}

		if (!std::getline(infile, line))
			break;
	}
	//Way of reordering!
	for (int i = 0; i < colors.size(); i++) {
		m_colors.push_back(colors.at(i));
	}
}

