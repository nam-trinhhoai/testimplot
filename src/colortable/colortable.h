#ifndef COLORTABLE_H
#define COLORTABLE_H

#include <QMetaType>

#include <string>
#include <vector>
#include <array>

class ColorTable
{
public:
	ColorTable();
	ColorTable(const  ColorTable & );
	ColorTable& operator=(
				const ColorTable&);
	bool operator==(
			const ColorTable&);

	bool readFile(const std::string &path);
	inline const std::string & getName() const
	{
		return m_name;
	}
	inline const std::array<int, 4>&  getColors(int i) const
	{
		return m_colors[i];
	}

	void  setAlpha(int i, int alpha)
	{
		 m_colors[i][3]=alpha;
	}

	int size() const
	{
		return m_colors.size();
	}

protected:
	std::string m_name;
	std::vector<std::array<int, 4>> m_colors;

};


Q_DECLARE_METATYPE(ColorTable)

#endif // COLORTABLE_H
