#ifndef COLORTABLEREGISTRY_H
#define COLORTABLEREGISTRY_H

#include <QMetaType>

#include <string>
#include <map>
#include <array>

#include "colortable.h"


class ColorTableRegistry {
public:
	ColorTableRegistry();
	static ColorTableRegistry & PALETTE_REGISTRY() {
		static ColorTableRegistry r;
		return r;
	}

	static ColorTable DEFAULT() {
		return PALETTE_REGISTRY().findColorTable("CLASSIC", "Black-White");
	}
	bool build(const std::string& path);

	int countFamillies();

	inline const std::map<const std::string, std::vector<ColorTable> > & getFamilies() {
		return m_paletteFamilies;
	}

	ColorTableRegistry(ColorTableRegistry const&) = delete;
	void operator=(ColorTableRegistry const&) = delete;

	ColorTable findColorTable(const std::string & familly,
			const std::string & name);

private:
	bool parseFamilly(const std::string& path);

private:
	std::map<const std::string, std::vector<ColorTable> > m_paletteFamilies;
};


#endif // COLORTABLE_H
