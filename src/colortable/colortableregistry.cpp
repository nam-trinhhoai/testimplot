#include "colortableregistry.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <iostream>

namespace fs = boost::filesystem;


ColorTableRegistry::ColorTableRegistry() {}

bool ColorTableRegistry::build(const std::string& path) {
	fs::path full_path(path);
	if (!fs::exists(full_path) || !fs::is_directory(full_path)) {
		std::cerr << "Color Tables Path Not found: " << full_path.native()
				<< std::endl;
		return false;
	}

	std::cout << "Color Tables found in directory: " << full_path.native()
			<< std::endl;
	fs::directory_iterator it(path);

	while (it != fs::directory_iterator { }) {

		try {
			if (fs::is_directory(*it))
				parseFamilly(it->path().string());

		} catch (const std::exception & ex) {
			std::cerr << "Failed to collect color tables in" << *it << " "
					<< ex.what() << std::endl;
		}
		*it++;
	}
}
bool ColorTableRegistry::parseFamilly(const std::string& path) {
	fs::path full_path(path);
	if (!fs::exists(full_path) || !fs::is_directory(full_path))
		return false;

	const std::string name = full_path.filename().string();
	fs::directory_iterator it(path);
	while (it != fs::directory_iterator { }) {
		try {
			ColorTable table;
			std::string current=it->path().string();
			if (!fs::is_directory(*it) && table.readFile(current))
				m_paletteFamilies[name].push_back(table);
			else
				std::cerr<<"Failed to read Color Table "<<current<<std::endl;

		} catch (const std::exception & ex) {
			return false;
		}
		*it++;
	}
	return true;
}

int ColorTableRegistry::countFamillies() {
	return m_paletteFamilies.size();
}

ColorTable ColorTableRegistry::findColorTable(const std::string & familly,
		const std::string & name) {
	const std::map<const std::string, std::vector<ColorTable> > palettes =
			ColorTableRegistry::PALETTE_REGISTRY().getFamilies();

	std::map<const std::string, std::vector<ColorTable> >::const_iterator it =
			palettes.find(familly);

	std::vector<ColorTable>::const_iterator itCol = it->second.begin();
	while (itCol != it->second.end()) {

		if (itCol->getName() == name)
			return *itCol;
		itCol++;
	}
	return palettes.begin()->second.front();

}


