
#ifndef __WELLUTIL__
#define __WELLUTIL__

#include <QColor>

#include <vector>
#include <ProjectManagerNames.h>

typedef struct _WELLBOREDATA
{
	QString tinyName;
	QString fullName;
	QString deviationTinyName;
	QString deviationFullName;
	ProjectManagerNames logs;
	ProjectManagerNames tf2p;
	ProjectManagerNames picks;
}WELLBOREDATA;



typedef struct _WELLHEADDATA
{
	QString tinyName;
	QString fullName;
	std::vector<WELLBOREDATA> bore;
}WELLHEADDATA;



// for the basket
typedef struct _WELLBORELIST
{
	QString bore_tinyname;
	QString bore_fullname;
	QString deviation_fullname;
	std::vector<QString> log_tinyname;
	std::vector<QString> log_fullname;
	std::vector<QString> log_displayname;
	std::vector<QString> tf2p_tinyname;
	std::vector<QString> tf2p_fullname;
	std::vector<QString> tf2p_displayname;
	std::vector<QString> picks_tinyname;
	std::vector<QString> picks_fullname;
	std::vector<QString> picks_displayname;
}WELLBORELIST;


typedef struct _WELLLIST
{
	QString head_tinyname;
	QString head_fullname;
	std::vector<WELLBORELIST> wellborelist;
}WELLLIST;

typedef struct _WELLMASTER
{
	ProjectManagerNames m_basketWell;
	ProjectManagerNames m_basketWellBore;
	std::vector<WELLLIST> finalWellBasket;
}WELLMASTER;





// picks sorted wells
typedef struct _WELLBOREPICKSLIST
{
	QString boreName;
	QString borePath;
	QString deviationPath;
	QString picksName;
	QString picksPath;
}WELLBOREPICKSLIST;


typedef struct _WELLPICKSLIST
{
	QString name;
	QString path;
	std::vector<WELLBOREPICKSLIST> wellBore;
}WELLPICKSLIST;

typedef struct _MARKER
{
	QString name;
	QColor color;
	std::vector<WELLPICKSLIST> wellPickLists;
} MARKER;



class WellUtil
{
	public :
		//static void getIndexFromWellWellbore(const std::vector<WELLHEADDATA>& data, QString txt, int *idx_well, int *idx_bore);
		static void getIndexFromWellBoreFullname(const std::vector<WELLHEADDATA>& data, QString fullname, int *idx_well, int *idx_bore);
		static bool getIndexFromWellLists(const std::vector<WELLLIST>& data, QString fullname, int *idx_well, int *idx_bore);
		static void wellListCreateGetIndex(std::vector<WELLLIST> *well_list, QString head_tinyname, QString head_fullname,
				QString bore_tinyname, QString bore_fullname,
				QString deviation_fullname,
				int *idx_well, int *idx_bore);

		static std::vector<std::vector<QString>> multiCriterionSearchFormat(const QString& str);
		static int qstring_display_listPrefix_valid(QString str, std::vector<QString> prefix);
		static int well_wellbore_display_valid(const std::vector<WELLHEADDATA>& data0, const QString& prefix, int i_well, int i_bore);


};

#endif
