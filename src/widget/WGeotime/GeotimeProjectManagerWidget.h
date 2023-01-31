/*
 * 
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */

#ifndef MURATAPP_SRC_TOOLS_XCOM_GEOTIMEPROJECTMANAGERWIDGET_H_
#define MURATAPP_SRC_TOOLS_XCOM_GEOTIMEPROJECTMANAGERWIDGET_H_

#include <vector>

#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QListWidget>
#include <QDir>
#include <QLineEdit>
#include <QTabWidget>
#include <QGroupBox>
#include <QTableWidget>
#include <QPushButton>

#include <vector>
#include <math.h>

#include <picksmanager.h>

#include "utils/stringutil.h"
#include <boost/filesystem.hpp>
#include <string.h>
#include <WellUtil.h>
// #include "GeotimeProjectManagerWidget.h"

class QTableView;
class QStandardItemModel;

/*
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
*/



// proto
typedef struct _PMANAGER_NAMES
{
	QString fullname;
	QString tinyname;
}PMANAGER_NAMES;

typedef struct _PMANAGER_LISTNAMES
{
	std::vector<QString> fullname;
	std::vector<QString> tinyname;
}PMANAGER_LISTNAMES;

typedef struct _PMANAGER_WELLBORE
{
	PMANAGER_NAMES bore;
	std::vector<PMANAGER_NAMES> logs;
	std::vector<PMANAGER_NAMES> tf2p;
	std::vector<PMANAGER_NAMES> picks;
	PMANAGER_NAMES deviation;
}PMANAGER_WELLBORE;

typedef struct _PMANAGER_WELL
{
	PMANAGER_NAMES head;
	std::vector<PMANAGER_WELLBORE> wellbore;
}PMANAGER_WELL;

typedef struct _PMANAGER_CULTURAL
{
	std::vector<QString> cdata_tinyname;
	std::vector<QString> cdata_fullname;
	std::vector<QString> strd_tinyname;
	std::vector<QString> strd_fullname;
}PMANAGER_CULTURAL;

typedef struct _PMANAGER_DATA
{
	int project_type;
	PMANAGER_NAMES project;
	PMANAGER_NAMES survey;

	std::vector<PMANAGER_NAMES> seismic;
	std::vector<PMANAGER_NAMES> horizons;
	PMANAGER_CULTURAL culturals;
	std::vector<PMANAGER_NAMES> neurons;
	std::vector<PMANAGER_WELL> well;
	PMANAGER_CULTURAL cultural;
}PMANAGER_DATA;

// display


typedef struct _PMANAGER_BORE_DISPLAY
{
	QString bore_tinyname;
	QString bore_fullname;
	QString deviation_fullname;
	std::vector<QString> log_tinyname;
	std::vector<QString> log_fullname;
	std::vector<QString> tf2p_tinyname;
	std::vector<QString> tf2p_fullname;
	std::vector<QString> picks_tinyname;
	std::vector<QString> picks_fullname;
}PMANAGER_BORE_DISPLAY;




typedef struct _PMANAGER_WELL_DISPLAY
{
	/*
	PMANAGER_NAMES well;
	PMANAGER_NAMES wellbore;
	PMANAGER_NAMES name;
	*/
	QString head_tinyname;
	QString head_fullname;
	std::vector<PMANAGER_BORE_DISPLAY> bore;
}PMANAGER_WELL_DISPLAY;


typedef struct _PMANAGER_DISPLAY
{
	int project_type;
	std::vector<PMANAGER_LISTNAMES> project_main_list;
	// std::vector<PMANAGER_NAMES> project;
	std::vector<PMANAGER_NAMES> survey;
	std::vector<PMANAGER_NAMES> seismic;
	std::vector<PMANAGER_NAMES> horizons;
	PMANAGER_CULTURAL culturals;
	std::vector<PMANAGER_NAMES> neurons;

	std::vector<PMANAGER_WELL_DISPLAY> wells;

	PMANAGER_CULTURAL cultural;
}PMANAGER_DISPLAY;










class GeotimeProjectManagerWidget : public QWidget{
	Q_OBJECT
public:
	GeotimeProjectManagerWidget(QWidget* parent = 0);
	virtual ~GeotimeProjectManagerWidget();
	QString projectpath, surveypath, wellpath;
	QString get_well_directory(QString projectdir, QString projectname);

	void removeTabSeismic();
	void removeTabHorizons();
	void removeTabCulturals();
	void removeTabWells();
	void removeTabNeurons();
	void removeTabPicks();
	void removeTabNurbs();
	bool isTabSeismic();
	bool isTabHorizons();
	bool isTabCulturals();
	bool isTabWells();
	bool isTabNeurons();
	bool isTabPicks();
	bool isTabNurbs();

	QString getNextVisionPath();
	QString getNVHorizonPath();
	QString getIsoHorizonPath();
	QString getNextVisionSeismicPath();
	QString getPatchPath();
	QString getVideoPath();


	QString get_projet_name();
	QString get_projet_fullpath_name();
	QString get_survey_name();
	QString get_survey_fullpath_name();
	QString get_nurbs_path0();


	std::vector<QString> get_seismic_names();
	std::vector<QString> get_seismic_AllTinynames();// MZR27082021
	std::vector<QString> get_seismic_AllFullnames(); // MZR 30082021
//	std::vector<QString> get_seismic_fullnames();
	std::vector<QString> get_seismic_fullpath_names();
	std::vector<QString> get_freehorizon_names();
	std::vector<QString> get_freehorizon_fullnames();
	std::vector<QString> get_isohorizon_names();
	std::vector<QString> get_isohorizon_fullnames();
	
	std::vector<QString> get_nurbs_names0();
	std::vector<QString> get_nurbs_fullnames0();


	std::vector<QString> get_horizonanim_names0();
	std::vector<QString> get_horizonanim_fullnames0();



	std::vector<QString> get_nurbs_names();
	std::vector<QString> get_nurbs_fullnames();

	std::vector<QString> get_horizon_names();
	std::vector<QString> get_horizon_fullnames();
	std::vector<QString> get_horizon_fullpath_names();

	std::vector<QString> get_rgb_names();
	std::vector<QString> get_rgb_fullnames();

	// std::vector<QString> get_cultural_names();
	// std::vector<QString> get_cultural_fullnames();
	// std::vector<QString> get_cultural_fullpath_names();

	std::vector<QString> get_wells_names();
	std::vector<QString> get_wells_fullnames();
	std::vector<QString> get_wells_fullpath_names();	

	std::vector<QString> get_neurons_names();
	std::vector<QString> get_neurons_fullnames();
	std::vector<QString> get_neurons_fullpath_names();

	std::vector<QString> get_culturals_cdat_names();
	std::vector<QString> get_culturals_cdat_fullnames();

	std::vector<QString> get_culturals_strd_names();
	std::vector<QString> get_culturals_strd_fullnames();

	std::vector<QString> get_picks_names();
	std::vector<QString> get_picks_fullnames();
	std::vector<QBrush> get_picks_colors();
	std::vector<QString> get_all_picks_names();
	std::vector<QString> get_all_picks_fullnames();
	std::vector<QBrush> get_all_picks_colors();

	std::vector<QString> get_freehorizon_names_basket();
	std::vector<QString> get_freehorizon_fullnames_basket();

	std::vector<QString> get_isohorizon_names_basket();
	std::vector<QString> get_isohorizon_fullnames_basket();

	std::vector<QString> get_horizonanim_names_basket();
	std::vector<QString> get_horizonanim_fullnames_basket();


//	std::vector<QString> get_welllog_names();
//	std::vector<QString> get_wellog_fullpath_names();

//	std::vector<QString> get_welltfp_names();
//	std::vector<QString> get_welltfp_fullpath_names();

//	std::vector<QString> get_wellpicks_names();
//	std::vector<QString> get_wellpicks_fullpath_names();

//	std::vector<QString> get_welldeviation_fullpath_names();


	int getIndexFromVectorString(std::vector<QString> list, QString txt);

	void setCulturalsVisible(bool val);
	void setNeuronsVisible(bool val);
	void setWellsVisible(bool val);
	void setHorizonsVisible(bool val);
	void setSeismicVisible(bool val);

	QString get_horizons_path0();
	QString get_seismic_path0();

	void load_session(QString sessionPath);
	void save_session(QString sessionPath);
	void load_session_gui();
	void save_session_gui();
	void load_last_session();


	void fill_empty_logs_list();
	void seismic_database_update();
	void well_database_update();
	void pick_database_update();



// <<<<<<< HEAD
	PMANAGER_DATA data0;
	PMANAGER_DISPLAY display0;

	std::vector<WELLLIST> well_list;
// =======
// >>>>>>> be94b96fff0582132ec38ff735b99c4e40b1dbd7
	std::vector<WELLLIST> get_well_list();
	std::vector<PMANAGER_WELL_DISPLAY> get_display_well_list();
	static 	int filext_axis(QString filename);

	void seismic_names_database_update();

	void global_seismic_database_update();
	void global_rgb_database_update();
	QString get_ImportExportPath();
	QString get_cubeRGT2RGBPath();
	QString get_IJKPath();


	static QString get_deviation_fullname(QString wellbore_fullname);
	std::vector<MARKER> getPicksSortedWells();
	std::vector<MARKER> staticGetPicksSortedWells(const std::vector<QString>& picksName,
			const std::vector<QBrush>& colors, const std::vector<PMANAGER_WELL_DISPLAY>& wellsList0);

	QString get_horizonanim_path0();
	void set_wells_filter(const QString& filter);
	bool select_well_fullpath_name(const QString& searched_fullpath_name);
	void clear_well_gui_selection();
	bool select_well_basket_fullpath_name(const QString& searched_fullpath_name);
	void clear_well_basket_gui_selection();

	bool add_pick_fullpath_name(const QString& pick_fullpath_name);
	bool remove_pick_fullpath_name(const QString& pick_fullpath_name);

	void set_seismics_filter(const QString& filter);
	bool select_seismic_tinyname(const QString& searched_tinyname);
	void clear_seismic_gui_selection();
	bool select_seismic_basket_tinyname(const QString& searched_fullpath_name);
	void clear_seismic_basket_gui_selection();

	void set_tfps_filter(const QString& filter);
	bool select_well_basket_tinyname(const QString& well_name);
	void select_all_tfp_basket();
	bool select_tfp_tinyname(const QString& tfp_name);

	void add_freehorizon(const QString& freehorizon_fullname, const QString& freehorizon_tinyname);
	void remove_freehorizon(const QString& freehorizon_fullname);

	void add_isohorizon(const QString& isohorizon_fullname, const QString& isohorizon_tinyname);
	void remove_isohorizon(const QString& isohorizon_fullname);

	void add_horizonanim(const QString& horizonanim_fullname, const QString& horizonanim_tinyname);
	void remove_horizonanim(const QString& horizonanim_fullname);

	void set_nurbs_filter(const QString& filter);
	bool select_nurbs_tinyname(const QString& searched_fullpath_name);
	void clear_nurbs_gui_selection();
	bool select_nurbs_basket_tinyname(const QString& searched_fullpath_name);
	void clear_nurbs_basket_gui_selection();

	bool is_session_loaded(); // check that a session has been loaded and the project_type, project and survey are the same
	bool save_to_default_session(bool force=false);


public slots:
	void trt_well_basket_add();
	void trt_well_basket_sub();
	void trt_seismic_basket_add();
    void trt_seismic_basket_sub();
    void trt_welltf2p_basket_add();
    void trt_welltf2p_basket_sub();
    void trt_nurbs_basket_add();
    void trt_nurbs_basket_sub();
    void trt_nurbs_database_update();

private:
	QFileInfoList survey_real_list;
	QComboBox *cb_projecttype;
	QListWidget *lw_projetlist, *lw_surveylist,
	*lw_cultural, *lw_cultural_basket, *lw_wells, *lw_wellsbasket, *lw_neurons, *lw_horizons, *lw_seismic, *lw_wellbore,
	*lw_seismic_basket, *lw_horizons_basket, *lw_nurbs, *lw_nurbs_basket,
	*qlw_welllog, *qlw_welllog_basket, *qlw_welltf2p, *qlw_welltf2p_basket, *qlw_wellpicks, *qlw_wellpicks_basket,
	*qlw_rgb, *qlw_rgb_basket;
	QLineEdit *lineedit_projectsearch, *lineedit_surveysearch, *lineedit_culturalsearch, *lineedit_wellssearch, *linedit_wellboresearch,
		*lineedit_neuronssearch, *lineedit_horizonssearch, *lineedit_seismicsearch,*lineedit_nurbssearch,
		*linedit_welllogsearch, *linedit_welltf2psearch, *linedit_wellpickssearch,
		*lineedit_rgt2rgtsearch,
		*lineedit_custompath;
	QCheckBox *chkbx_culturals, *chkbx_wells, *chkbx_neurons, *chkbx_horizons;
	QTabWidget *tabw_table1;
	QLabel *label_projectname, *label_surveyname, *label_projecttype, *label_custom_path;
	QGroupBox* qgb_projectmanager;
	QTableWidget *qtw_welllog_basket;
	QPushButton *qpb_custompath;


	// std::vector<WELLLIST> well_list;

	// QString *array_survey_name0;

	std::vector<QString> survey_dirname;
	std::vector<QString> survey_name;

	std::vector<QString> m_seismic_fullpath;
	std::vector<QString> m_seismic_fullpath_basket;
	std::vector<QString> seismic_fullname;
	std::vector<QString> seismic_tinyname;
	std::vector<QBrush> seismic_diplay_color;
	std::vector<QString> seismic_tinyname_basket;
	std::vector<QBrush> seismic_basket_color;

	std::vector<QString> horizons_fullname;
	std::vector<QString> horizons_tinyname;
	std::vector<QString> horizons_tinyname_basket;
	std::vector<QString> horizons_fullname_basket;	

	// QString current_well_head;
	// QString current_well_bore;

	std::vector<QString> wells_maindir_fullname;
	std::vector<QString> wells_maindir_tinyname;

	std::vector<QString> well_head_fullname;
	std::vector<QString> well_head_tinyname;
	std::vector<QString> well_bore_fullname;
	std::vector<QString> well_bore_tinyname;
	std::vector<QString> well_wellbore_tinyname;


	std::vector<QString> well_bore;

	std::vector<QString> well_head_tinyname_basket;
	std::vector<QString> well_head_fullname_basket;
	std::vector<QString> well_bore_tinyname_basket;
	std::vector<QString> well_bore_fullname_basket;
	std::vector<QString> well_wellbore_basket;

	std::vector<QString> wells_fullname;
	std::vector<QString> wells_tinyname;

	std::vector<QString> wellslog_fullname;
	std::vector<QString> wellslog_tinyname;
	std::vector<QString> wellsft2p_fullname;
	std::vector<QString> wellsft2p_tinyname;
	std::vector<QString> wellspicks_fullname;
	std::vector<QString> wellspicks_tinyname;
	std::vector<QString> deviation_fullname;
	std::vector<QString> deviation_tinyname;

	std::vector<QString> wellslog_basket_fullname;
	std::vector<QString> wellslog_basket_tinyname;
	std::vector<QString> wellsft2p_basket_fullname;
	std::vector<QString> wellsft2p_basket_tinyname;
	std::vector<QString> wellspicks_basket_fullname;
	std::vector<QString> wellspicks_basket_tinyname;

	std::vector<QString> cultural_fullname;
	std::vector<QString> cultural_tinyname;

	std::vector<QString> neurons_fullname;
	std::vector<QString> neurons_tinyname;

	std::vector<QString> rgb_fullname;
	std::vector<QString> rgb_tinyname;
	std::vector<QString> rgb_basket_fullname;
	std::vector<QString> rgb_basket_tinyname;

	std::vector<QString> nurbs_fullname;
	std::vector<QString> nurbs_tinyname;

	std::vector<QString> nurbs_tinyname_basket;
	std::vector<QString> nurbs_fullname_basket;

	std::vector<QString> freehorizons_tinyname_basket;
	std::vector<QString> freehorizons_fullname_basket;

	std::vector<QString> isohorizons_tinyname_basket;
	std::vector<QString> isohorizons_fullname_basket;

	std::vector<QString> horizonanims_tinyname_basket;
	std::vector<QString> horizonanims_fullname_basket;

	// to update when loading default session (loaded session become default session with load_session_gui)
	// to update when saving default session (saved session become default session with save_session_gui)
	QString m_session_project_type_cache;
	QString m_session_use_path_cache;
	QString m_session_project_cache;
	QString m_session_survey_cache;
	QString m_session_path_cache;

	bool m_session_loaded = false;
	void init_session_cache();


	bool tab_seismic, tab_horizons, tab_wells, tab_culturals, tab_neurons, tab_picks, tab_nurbs;
	void removeTab(QString name);

	QFileInfoList get_seismic_list(QString path);
	std::vector<QString> get_seismic_list2(QString path);
	QFileInfoList get_horizons_list(QString path);
	// QFileInfoList get_cultural_list(QString path);
	QFileInfoList get_neurons_list(QString path);

	std::vector<QString> get_nurbs_list(QString path);
	std::vector<QString> get_horizonanim_list(QString path);

	QString get_project_path(int project_type);
	QString get_survey_path(int project_type, QString project_name);
	QString get_seismic_path(int project_type, QString project_name, QString survey_name);
	QString get_horizons_path(int project_type, QString project_name, QString survey_name);
	QString get_wells_path(int project_type, QString project_name);
	QString get_cultural_path(int project_type, QString project_name);
	QString get_nurbs_path(int project_type, QString project_name, QString survey_name);

	QString get_project_path0();
	QString get_survey_path0();
	// QString get_horizons_path0();
	QString get_wells_path0();
	QString get_cultural_path0();
	QString get_neurons_path0();
	


	void display_label_titles();
	void display_project_list(int project_type, QString prefix);
	void display_survey_list(QString prefix);
	void display_seismic_list(QString prefix);	
	void display_wells_list(QString prefix);
	void display_horizons_list(QString prefix);
	void display_cultural_list(QString prefix);
	void display_neurons_list(QString prefix);
	void display_nurbs_list(QString prefix);

	void survey_names_update();

	void seismic_names_database_create();
	void seismic_names_disk_update();
	void seismic_names_update();

	void horizon_names_database_update();
	void horizon_names_database_create();
	void horizon_names_disk_update();
	void horizon_names_update();

	void wells_names_update();
	void wells_names_disk_update();
	void wells_names_database_update();
	void wells_names_database_create();

	void cultural_names_database_create();
	void cultural_names_database_update();
	void cultural_names_disk_update();
	void cultural_names_update();

	void neurons_names_update();
	void welllog_names_update(QString path, int n_well, int n_bore);
	void welltf2p_names_update(QString path, int n_well, int n_bore);
	void wellpicks_names_update(QString path, int n_well, int n_bore);
	void welldeviation_names_update(QString path, int n_well, int n_bore);

	void welllog_names_update0(QString path);
	void welltf2p_names_update0(QString path);
	void wellpicks_names_update0(QString path);
	void welldeviation_names_update0(QString path);

    void rgb_names_update();
    void display_rgb_list(QString prefix);
    void rgb_names_disk_update();
    std::vector<QString> get_rgb_list(QString path);

    void nurbs_names_disk_update();


	void seismic_basket_add();
	void seismic_basket_sub();
	void display_seismic_basket_list();
	void horizons_basket_add();
	void horizons_basket_sub();
	void display_horizons_basket_list();	
	void display_nurbs_basket_list();

	void rgb_basket_add();
	void rgb_basket_sub();
	void display_rgb_basket_list();

	void nurbs_basket_add();
	void nurbs_basket_sub();

	void display_well_basket_list();
	void display_welllog(QString path, QString prefix);
	void display_welltf2p(QString path, QString prefix);
	void display_wellpicks(QString path, QString prefix);
	void display_welllog_basket_list();
	void display_welltf2p_basket_list();
	void display_wellpicks_basket_list();

	void clear_wells_data(int type);

	QFileInfoList get_dirlist(QString path);
	// std::vector<char*> get_dirlist2(QString path);
	std::vector<QString> get_dirlist2(QString path);

	// std::vector<char*> project_list[4];
	std::vector<std::vector<QString>> project_list;

	void welllist_create_get_index(QString head_tinyname, QString head_fullname,
			QString bore_tinyname, QString bore_fullname,
			QString deviation_fullname,
			int *idx_well, int *idx_bore);
	void welllist_get_index(QString head_tinyname, QString head_fullname,
			QString bore_tinyname, QString bore_fullname,
			int *idx_well, int *idx_bore);
	void welllist_get_index_from_logname(QString log_displayname, int *idx_well, int *idx_bore, int *idx);
	void welllist_get_index_from_tf2pname(QString displayname, int *idx_well, int *idx_bore, int *idx);
	void welllist_get_index_from_picksname(QString displayname, int *idx_well, int *idx_bore, int *idx);
	void welllist_get_index_from_borename(QString name, int *idx_well, int *idx_bore);

	//
	void welllist_get_index_from_wellbore_fullname(QString fullname, int *idx_well, int *idx_bore);




	QFileInfoList get_cultural_cdata_list(QString path);
	QFileInfoList get_cultural_strd_list(QString path);

	// update
	// int filext_axis(QString filename);
	std::vector<std::vector<QString>> multiCriterionSearchFormat(QString str);
	int well_wellbore_display_valid(QString prefix, int i_well, int i_bore);

	int qstring_display_valid(QString str, QString prefix);
	int qstring_display_listPrefix_valid(QString str, std::vector<QString> prefix);

	void project_list_init();
	std::vector<QString> get_directories(QString path);
	void project_list_display(QString prefix);
	QString get_survey_subdirectory();
	void display_culturals_basket_list();

	void getIndexFromWellWellboreFull(QString txt, int *idx_well, int *idx_bore);
	QString getProjIndexNameForDataBase();
	QString get_well_database_name();
	QString get_cultural_database_name();
	QString get_seismic_database_name();
	QString get_horizons_database_name();

	void trt_wellbasketlistclick(QString well_tiny_name);
	void memoryRAZ();
	void well_tf2p_clear();
	QString  getWellsHeadBasketSelectedName();

	// clear function for baskets, data lists and refresh, similar to memoryRAZ()
	void clear_project_specific_lists();
	void clear_survey_specific_lists();

	PicksManager *m_picksManager = nullptr;


	QFileInfoList getFiles(std::string path, QStringList ext);

private slots:
	void trt_projecttypeclick(int idx);
	void trt_projetlistClick(QListWidgetItem* p);
	void trt_projectsearchchange(QString str);
	void trt_surveylistClick(QListWidgetItem *p);
	void trt_surveysearchchange(QString str);
	void trt_wellssearchchange(QString str);
	void trt_horizonssearchchange(QString str);
	void trt_seismicsearchchange(QString str);
	void trt_culturalssearchchange(QString str);
	void trt_neuronssearchchange(QString str);
	void trt_welllistclick(QListWidgetItem *p);
	void trt_welllogsearchchange(QString str);
	void trt_welltf2psearchchange(QString str);
	void trt_wellpickssearchchchange(QString str);
	void trt_wellbasketlistclick(QListWidgetItem *p);
	void trt_wellbasketlistselectionchanged();
	void trt_rgbsearchchange(QString str);
	void trt_culturals_basket_add();
	void trt_culturals_basket_sub();

	void trt_chkbx1(int val);

	void trt_horizons_basket_add();
    void trt_horizons_basket_sub();
    void trt_welllog_basket_add();
    void trt_welllog_basket_sub();
    void trt_wellpicks_basket_add();
    void trt_wellpicks_basket_sub();
    void trt_rgb_basket_add();
    void trt_rgb_basket_sub();

    void trt_cultural_database_update();
    void trt_well_database_update();
    void trt_horizon_database_update();
    void trt_seismic_database_update();

    void trt_lineedit_custompath_return();
    void trt_custompath_valid();

    void trt_debug();

    QString format_dir_path(const QString& path_to_format);

    // void trt_ok();
    // void trt_cancel();
	// void computeZvsRho();
};

static const QString projectTypeKey = QStringLiteral("projectType");
static const QString projectPathKey = QStringLiteral("projectPath");
static const QString projectKey = QStringLiteral("project");
static const QString surveyKey = QStringLiteral("survey");
static const QString seismicKey = QStringLiteral("seismic");
static const QString horizonKey = QStringLiteral("horizon");
static const QString isoHorizonKey = QStringLiteral("rgt_iso");
static const QString culturalKey = QStringLiteral("cultural");
static const QString wellKey = QStringLiteral("well");
static const QString wellPathKey = QStringLiteral("wellPath");
static const QString wellLogKey = QStringLiteral("wellLogs");
static const QString wellTFPKey = QStringLiteral("wellTFP");
//static const QString wellPickKey = QStringLiteral("wellPicks");
static const QString nurbsKey = QStringLiteral("nurbs");
static const QString horizonAnimKey = QStringLiteral("anim");
static const QString neuronKey = QStringLiteral("neuron");
static const QString picksNamesKey = QStringLiteral("picksNames");
static const QString picksPathKey = QStringLiteral("picksPath");

static const QString tinynameKey = QStringLiteral("tinyname");
static const QString fullnameKey = QStringLiteral("fullname");

#endif 
