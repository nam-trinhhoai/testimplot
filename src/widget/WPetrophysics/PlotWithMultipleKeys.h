/*
 *
 *
 *  Created on: 21 Sept 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTWITHMULTIPLEKEYS_H_
#define NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTWITHMULTIPLEKEYS_H_

#include "QtImGuiCore.h"
#include "marker.h"
#include "workingsetmanager.h"
#include "DataSelectorDialog.h"
#include "geotimegraphicsview.h"
#include "folderdata.h"
#include "wellhead.h"
#include "wellbore.h"
#include "wellpick.h"
#include "seismicsurvey.h"
#include "seismic3ddataset.h"
#include "viewutils.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"
#include "implot.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "implot_internal.h"
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <stdio.h>
#include <map>


#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else

#endif
typedef struct chart;
typedef struct log_data;
typedef struct well_bore_config;
typedef struct log_view;
//struct well_bore_config {
//	WellBore* ;
//	bool linkedDepthAxis;
//	bool isAcitive;
//};
const char* getWellUnit(WellUnit wellUnit);
void colorPicker(ImVec4* tagColor, std::string lname);
template<typename T>
inline T RandomRange(T min, T max);
inline ImVec4 RandomColor();
typedef struct log_view {
	QString log_name;
	ImVec4 color;
	chart* acitiveChart;
	std::map<WellBore*, log_data*> m_plogs;
	log_view(QString log_name, ImVec4 color) {
		this->log_name = log_name;
		this->color = color;
		acitiveChart = nullptr;
	}
	~log_view(){
		m_plogs.clear();
	}
	void addLogData(WellBore* wb, log_data* plog) {
		m_plogs.insert(std::pair<WellBore*, log_data* >(wb, plog));
	}
	void removeLog(WellBore* wb) {
		m_plogs.erase(wb);
	}
	log_data* findByWellBore(WellBore* wb) {
		return m_plogs.find(wb) != m_plogs.end() ? (log_data*)m_plogs.find(wb)->second : nullptr;
	}
	void update_chart_idx(chart* charts) {
		acitiveChart = charts;
	}
	void reset() { acitiveChart = nullptr; }
}log_view;
typedef struct log_data {
	WellBore* wellbore;
	log_view* logView;
	std::string sUnit;
	int log_index; // index of log in wellbore
	WellUnit cur_unit; // current unit of keys

	double* keys;
	double* attributes;
	double attr_max_value;
	double attr_min_value;
	long start;
	long end;
	float opacity;
	double fillValue;
	float thickness;
	bool isGlobalThickness;
	bool shaded;
	// dùng để hiển thị null value
	std::string nullvalue;
	int num_points; //number of points in the largest nonnull interval
	bool is_empty;
	bool is_init;
	log_data() {
		keys = nullptr;
		attributes = nullptr;
		wellbore = nullptr;
		log_index = 0;
		cur_unit = UNDEFINED_UNIT;
		start = 0;
		end = 0;
		logView = nullptr;
		shaded = false;
		num_points = 0;
		is_empty = false;
		is_init = false;
		attr_min_value = 0.0f;
		attr_max_value = 0.0f;
		thickness = 1.0f;
		opacity = 0.5f;
		isGlobalThickness = true;
	}
	~log_data() {
		delete[] keys;
		delete[] attributes;

	}
	void initialize_data() {
		is_init = true;
		wellbore->selectLog(log_index);
		Logs l = wellbore->currentLog();

		if (l.nonNullIntervals.size() == 0) {
			is_empty = true;
			return;
		}
		attr_max_value = wellbore->maxi();
		attr_min_value = wellbore->mini();
		fillValue = attr_min_value;
		start = l.nonNullIntervals.front().first;
		end = l.nonNullIntervals.back().second;
		num_points = end - start + 1;
		nullvalue = std::to_string(l.nullValue);
		keys = new double[num_points];
		attributes = new double[num_points];
		std::copy(l.attributes.begin() + start, l.attributes.begin() + end + 1, attributes);
		sUnit = l.sUnit.toStdString();
		is_empty = false;
	}

	void update(WellBore& wb, std::map<QString, log_view*>* m_logs, int idx) {
		wellbore = &wb;
		log_index = idx;
		QString log_name = wb.logsNames()[log_index];
		if (m_logs->find(log_name) != m_logs->end()) {
			this->logView = m_logs->find(log_name)->second;
		}
		else {
			this->logView = new log_view(log_name, RandomColor());
			m_logs->insert(std::pair<QString, log_view*>(log_name, this->logView));
		}
		
		this->logView->addLogData(this->wellbore, this);
	}


	//
	void update_keys_on_unit(WellUnit unit) {
		if (cur_unit == unit) {
			return;
		}
		wellbore->selectLog(log_index);
		Logs current_log = wellbore->currentLog();
		cur_unit = unit;
		for (int i = 0; i < num_points; i++) {
			bool ok = false;
			double md_val = NULL;
			if (unit == WellUnit::MD) {
				md_val = wellbore->getMdFromWellUnit(current_log.keys[i + start], current_log.unit, &ok);
			}
			else if (unit == WellUnit::TVD) {
				md_val = wellbore->getDepthFromWellUnit(current_log.keys[i + start], current_log.unit, SampleUnit::DEPTH, &ok);
			}
			else if (unit == WellUnit::TWT) {
				md_val = wellbore->getDepthFromWellUnit(current_log.keys[i + start], current_log.unit, SampleUnit::TIME, &ok);
			}
			this->keys[i] = ok ? md_val : NULL;
		}
	}
}processed_log;

typedef struct chart {
	WellBore* wellbore;
	bool autofit;
	std::vector<log_view*> v_listLogs;
	float attr_min, attr_max;
	float fill_line;
	double x_max, x_min, y_max, y_min;
	bool selected;
	chart() {
		//does v_listLogs need reserve? a cap on number of plot on one chart?
		v_listLogs.clear();
		attr_min = std::numeric_limits<double>::max();
		attr_max = std::numeric_limits<double>::lowest();
		fill_line = 0;
		wellbore = nullptr;

		x_max = 200;
		x_min = -200;
		y_min = 0;
		y_max = 2000;
		autofit = true;
		selected = false;
	}
	void chartLimitValues(double* x_min, double* x_max, double* y_min, double* y_max) {
		this->x_min = *x_min;
		this->x_max = *x_max;
		this->y_min = *y_min;
		this->y_max = *y_max;
	}

	void resetAll() {
		v_listLogs.clear();
	}
	void add_log(log_view* p_log) {
		if (v_listLogs.empty() || v_listLogs.size() == v_listLogs.max_size()) {
			v_listLogs.reserve(5);
		}
		v_listLogs.push_back(p_log);
		attr_min = std::min(attr_min, (float)p_log->findByWellBore(wellbore)->attr_min_value);
		attr_max = std::max(attr_max, (float)p_log->findByWellBore(wellbore)->attr_max_value);
		x_min = attr_min;
		x_max = attr_max;
	};
	void currentBore(WellBore* wellbore) {
		if (this->wellbore != wellbore) {
			this->wellbore = wellbore;
		}
	}
	void remove_log(log_view* p_log) {
		attr_min = std::numeric_limits<double>::max();
		attr_max = std::numeric_limits<double>::lowest();
		for (int i = 0; i < v_listLogs.size(); i++) {
			if (p_log == v_listLogs[i]) {
				v_listLogs.erase(v_listLogs.begin() + i);
			}
			else {
				attr_min = std::min(attr_min, (float)p_log->findByWellBore(wellbore)->attr_min_value);
				attr_max = std::max(attr_max, (float)p_log->findByWellBore(wellbore)->attr_max_value);
			}

		}
		fill_line = attr_min;
	}
	void calShadeRange() {
		float min = std::numeric_limits<double>::max();
		float max = std::numeric_limits<double>::lowest();
		for (int i = 0; i < v_listLogs.size(); i++) {
			min = std::min(min, (float)v_listLogs[i]->findByWellBore(wellbore)->attr_min_value);
			max = std::max(min, (float)v_listLogs[i]->findByWellBore(wellbore)->attr_min_value);

		}
		attr_min = min;
		attr_max = max;
		x_min = attr_min;
		x_max = attr_max;
	}
}chart;
typedef struct track_rule {
	char* wellUnit;
	int column;
	track_rule(int i, WellUnit wellUnit) {
		this->column = i;
		strcpy(this->wellUnit, getWellUnit(wellUnit));
	}
	void changeWellUnit(WellUnit wellUnit) {
		strcpy(this->wellUnit, getWellUnit(wellUnit));
	}

}track_rule;
class PlotWithMultipleKeys : public QtImGuiCore
{
public:
	PlotWithMultipleKeys(WorkingSetManager* manager);

	virtual ~PlotWithMultipleKeys();

	void showPlot() override;

private:
	char inputTag[20];
	typedef struct IJKPoint {
		int i;
		int j;
		int k;
	} IJKPoint;

	// Data manager
	QList<WellPick*> m_picks;
	WellPick* pick;

	int selectedWell = -1;
	WorkingSetManager* m_manager;
	// Etablish a list of seismic dataset from selected case study
	std::vector<Seismic3DAbstractDataset*> listSeismicDatasets;
	// Etablish a list of wellbores that have logs from selected case study
	FolderData* wells;
	FolderData* seismics;
	QList<IData*> iData;
	QList<IData*> iData_Seismic;

	// Chart Setting
	int selected_track = 0;
	double r_min = 0;
	double r_max = 2000;
	float r_limit = 2000;
	bool r_change = true;

	std::vector<WellBore*> listWellBores;
	//std::vector<std::string> logNames;
	std::map<QString, log_view*> m_logViews;

	bool* linkDepthAxis;
	bool useLongCrossHair = false;
	ImVec4 background_color;
	WellUnit selectedWellUnit = WellUnit::MD;
	ImGuiComboFlags flags = 0;
	float line_weight = 1;
	float chart_width = 300;
	int totalNumberOfWellBores;
	//Charts variable
	int total_logs_count;
	// Number of chart areas
	int numChartAreas;
	const int maxNumChart = 12;
	// processed log loaded
	log_data* processed_logs_ptr;
	// Number of depth points
	int numPoints;
	double tag_value;
	// Names of log curves

	float opacity = 0.5f;
	std::vector<chart*> charts;
	track_rule* trackRule;
	ImPlotRect* lims;
	// function
	void chartHeader();
	log_view* findLogViewByLogname(QString lname);
	bool showDepth();
	void showActiveLog(log_data* plog);
	void widgetStyle();
	void longCrossHairCursor();
	void setting(ImGuiStyle& style);
	void markerSetting();
	int interactiveHelper(double* depth);
	void removeLog(chart* chart, log_data* curLog);
	void addLogInChart(int idx, QString lname);
	void resetSelectedChart();
	void menubar(ImGuiStyle& style);
	std::pair<bool, IJKPoint> isPointInBoundingBox(Seismic3DAbstractDataset* dataset, WellUnit wellUnit, double logKey, WellBore* wellBore);
};

#endif // NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTWITHMULTIPLEKEYS_H_
