/*
 *
 *
 *  Created on: 21 Sept 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTWITHMULTIPLEKEYS_H_
#define NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTWITHMULTIPLEKEYS_H_

#include "QtImGuiCore.h"

#include "workingsetmanager.h"
#include "DataSelectorDialog.h"
#include "geotimegraphicsview.h"
#include "folderdata.h"
#include "wellhead.h"
#include "wellbore.h"
#include "seismicsurvey.h"
#include "seismic3ddataset.h"
#include "viewutils.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"

#include "ImGuiFileBrowser.h"
#include "implot.h"
#include "imgui.h"


#include "imgui_internal.h"

#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <stdio.h>


#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else

#endif

template<typename T>
inline T RandomRange(T min, T max);
inline ImVec4 RandomColor();

class PlotWithMultipleKeys : public QtImGuiCore
{
public:
	PlotWithMultipleKeys(WorkingSetManager* manager);
	struct processed_log {
		WellBore* wellbore;
		int log_index; // index of log in wellbore
		WellUnit init_unit; // initial unit of keys
		QString log_name;
		double* keys;
		double* attributes;
		long start;
		long end;
		ImVec4 color;
		int chart_idx;
		int num_points;
		processed_log() {
			keys = nullptr;
			attributes = nullptr;
			wellbore = nullptr;
			log_index = 0;
			init_unit = MD;
			start = 0;
			end = 0;
			chart_idx = -1;
			num_points = 0;
		}
		~processed_log() {
			delete[] keys;
			delete[] attributes;
		}
		void update(WellBore& wb, Logs& l, int idx) {
			wellbore = &wb;
			log_index = idx;
			init_unit = l.unit;
			log_name = wb.logsNames()[log_index];
			attributes = &l.attributes[0];
			num_points = l.attributes.size();
			keys = new double[num_points];
			attributes = new double[num_points];

			update_keys_on_unit(WellUnit::MD);
			update_attributes();
			WellBore::computeNonNullInterval(l);
			start = l.nonNullIntervals.front().first;
			end = l.nonNullIntervals.back().second;

			color = RandomColor();
			chart_idx = -1;
		}
		void update_attributes() {
			wellbore->selectLog(log_index);
			Logs current_log = wellbore->currentLog();
			for (int i = 0; i < num_points; i++) {
				this->attributes[i] = current_log.attributes[i];
			}
		}
		//
		void update_keys_on_unit(WellUnit unit) {
			wellbore->selectLog(log_index);
			Logs current_log = wellbore->currentLog();
			for (int i = 0; i < num_points; i++) {
				bool ok = false;
				double md_val = NULL;
				if (unit == WellUnit::MD) {
					md_val = wellbore->getMdFromWellUnit(current_log.keys[i], init_unit, &ok);
				}
				else if (unit == WellUnit::TVD) {
					md_val = wellbore->getDepthFromWellUnit(current_log.keys[i], init_unit, SampleUnit::DEPTH, &ok);
				}
				else if (unit == WellUnit::TWT) {
					md_val = wellbore->getDepthFromWellUnit(current_log.keys[i], init_unit, SampleUnit::TIME, &ok);
				}
				this->keys[i] = ok ? md_val : NULL;
			}
		}
		void reset() { chart_idx = -1; }

	};
	struct MyDndItem {
		int              Idx;
		int              Plt;
		int 			 chartIdx;
		ImVector<ImVec2> Data;
		ImVec4           Color;
		MyDndItem() {
			static int i = 0;
			Idx = i++;
			Plt = 0;
			chartIdx = 0;

			Color = RandomColor();

			// tmp solution to activate ImGuiPayLoad
			Data.reserve(2);
			for (int k = 0; k < 2; ++k) {
				float t = k;
				Data.push_back(ImVec2(t, Idx));
			}
		}
		void Reset() { Plt = 0; chartIdx = 0; }
	};

	virtual ~PlotWithMultipleKeys();

	void showPlot() override;
private:
	// Data manager
	WorkingSetManager* m_manager;
	FolderData* wells;
	FolderData* seismics;
	QList<IData*> iData;
	QList<IData*> iData_Seismic;
	bool linkDepthAxis = false;
	bool useLongCrossHair = false;
	WellUnit selectedWellUnit = WellUnit::MD;
	ImGuiComboFlags flags = 0;
	// Etablish a list of wellbores that have logs from selected case study
	std::vector<WellBore*> listWellBores;
	int totalNumberOfWellBores;
	// Etablish a list of seismic dataset from selected case study
	std::vector<Seismic3DAbstractDataset*> listSeismicDatasets;

	//Charts variable
	// Number of chart areas
	int total_logs_count;
	int numChartAreas;
	processed_log processed_logs[100];
	//	MyDndItem dnd[100]; // 100 is a limit number of logs
	//	MyDndItem dnd_seismic[100];// 100 is a limit number of seismic extractions

		// Number of depth points
	int numPoints;

	// Names of log curves
	std::vector<std::string> logNames;

	// Long crosshair cursor
	void longCrossHairCursor();
	void setting(ImGuiStyle& style);
	// Interactive helper
	int interactiveHelper(double* depth);

	typedef struct IJKPoint {
		int i;
		int j;
		int k;
	} IJKPoint;

	std::pair<bool, IJKPoint> isPointInBoundingBox(Seismic3DAbstractDataset* dataset, WellUnit wellUnit, double logKey, WellBore* wellBore);
};

#endif // NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PLOTWITHMULTIPLEKEYS_H_
