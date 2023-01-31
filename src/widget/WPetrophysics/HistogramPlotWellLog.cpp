/*
 *
 *
 *  Created on: 19 Jul 2022
 *      Author: l0359127
 */

#include "HistogramPlotWellLog.h"

HistogramPlotWellLog::HistogramPlotWellLog(WorkingSetManager* manager):
m_manager(manager)
{}

HistogramPlotWellLog::~HistogramPlotWellLog()
{}

// Cross-plot between the logs
void HistogramPlotWellLog::showPlot() {
	// We demonstrate using the full viewport area or the work area (without menu-bars, task-bars etc.)
	// Based on your use case you may want one of the other.
	static bool use_work_area = true;
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);
	ImGui::SetNextWindowSize(use_work_area ? viewport->WorkSize : viewport->Size);


	// Get data from database
	WorkingSetManager::FolderList folders = m_manager->folders();
	FolderData* wells = folders.wells;
	QList<IData*> iData = wells->data();

	// Etablish a list of wellbores that have logs from selected case study
	std::vector<WellBore*> listWellBores;

	int iDataSize = iData.size();

	if (iDataSize > 0)
	{
		for (int i=0; i<iDataSize; i++)
		{
			WellHead* wellHead = dynamic_cast<WellHead*>(iData[i]);

			int numberOfWellBores = wellHead->wellBores().size();

			if (numberOfWellBores > 0)
			{
				for (int iWellbore=0; iWellbore < numberOfWellBores; iWellbore++)
				{
					WellBore* bore = wellHead->wellBores()[iWellbore];

					bool hasLogs = (bore->logsNames().size() > 0);
					
					// Only add wellbores that have logs to the list
					if (hasLogs)
						listWellBores.push_back(bore);	
				}
			}
		}
	}	

	// Plot is available only if at least one wellbore has logs
	int totalNumberOfWellBores = listWellBores.size();
	if(totalNumberOfWellBores > 0)
	{
		static ImGuiWindowFlags winflags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;
		bool showPlot = true;

		ImGui::Begin("Cross-plot", &showPlot, winflags);
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
		ImGui::BeginChild("ChildLeft", ImVec2(ImGui::GetContentRegionAvail().x*0.2f, -1), false, window_flags);


		// Get data
		double *log1;
		
		std::string logName1 = " ";
		
		// Initialization
		WellBore *bore;
		bool logIsSelected;
		Logs currentLog1;

		if (ImGui::TreeNode("WellBores"))
		{
			// Select a well
			static int selectedWell = -1;
			
			//for (int n = 0; n < wellHead->wellBores().size(); n++)
			for (int n = 0; n < totalNumberOfWellBores; n++)
			{
				if (ImGui::Selectable(listWellBores[n]->name().toStdString().c_str(), selectedWell == n))
					selectedWell = n;
			}
			ImGui::TreePop();

			if(selectedWell >-1)
			{
				bore = listWellBores[selectedWell];
	
				// Get log data from selected well
				std::vector<std::string> logNames;
				for (int i=0; i<bore->logsNames().size();i++)
				{
					logNames.push_back( bore->logsNames()[i].toStdString() );	
				}

				int numLogs = bore->logsNames().size();

				// List of logs for selection
				{
					if (ImGui::TreeNode("Select the first log"))
					{
						static int selected = -1;
						for (int n = 0; n < numLogs; n++)
						{
							if (ImGui::Selectable(bore->logsNames()[n].toStdString().c_str(), selected == n))
								selected = n;
						}
						ImGui::TreePop();

						if(selected >-1)
						{
							logName1 = bore->logsNames()[selected].toStdString();

							logIsSelected = bore->selectLog(selected);

							currentLog1 = bore->currentLog();
							numPoints = currentLog1.attributes.size();

							log1 = &currentLog1.attributes[0];
						}
					}
				}
			}

		} // ImGui::TreeNode("Wells")

		ImGui::EndChild();

		ImGui::SameLine();

		// Plot histogram
		{
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
	
			//ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
			ImGui::BeginChild("ChildRight", ImVec2(-1, -1), true, window_flags);

			static int  bins       = 50;
			static bool cumulative = false;
			static bool density    = true;
			static bool outliers   = true;
		
			ImGui::SetNextItemWidth(200);
			if (ImGui::RadioButton("Sqrt",bins==ImPlotBin_Sqrt))       { bins = ImPlotBin_Sqrt;    } ImGui::SameLine();
			if (ImGui::RadioButton("Sturges",bins==ImPlotBin_Sturges)) { bins = ImPlotBin_Sturges; } ImGui::SameLine();
			if (ImGui::RadioButton("Rice",bins==ImPlotBin_Rice))       { bins = ImPlotBin_Rice;    } ImGui::SameLine();
			if (ImGui::RadioButton("Scott",bins==ImPlotBin_Scott))     { bins = ImPlotBin_Scott;   } ImGui::SameLine();
			if (ImGui::RadioButton("N Bins",bins>=0))                       bins = 50;
			if (bins>=0) {
				ImGui::SameLine();
				ImGui::SetNextItemWidth(200);
				ImGui::SliderInt("##Bins", &bins, 1, 100);
			}
			if (ImGui::Checkbox("Density", &density))
			{
				ImPlot::SetNextAxisToFit(ImAxis_X1);
				ImPlot::SetNextAxisToFit(ImAxis_Y1);
			}
			ImGui::SameLine();
			if (ImGui::Checkbox("Cumulative", &cumulative))
			{
				ImPlot::SetNextAxisToFit(ImAxis_X1);
				ImPlot::SetNextAxisToFit(ImAxis_Y1);
			}
			ImGui::SameLine();
			static bool range = false;
			ImGui::Checkbox("Range", &range);
			static float rmin = -3;
			static float rmax = 13;
			if (range) {
				ImGui::SameLine();
				ImGui::SetNextItemWidth(200);
				ImGui::DragFloat2("##Range",&rmin,0.1f,-3,13);
				ImGui::SameLine();
				ImGui::Checkbox("Outliers",&outliers);
			}

			if (ImPlot::BeginPlot("##Histograms", ImVec2(-1, -1))) {
				ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);

				if (currentLog1.attributes.size()>0)
					ImPlot::PlotHistogram(logName1.c_str(), log1, numPoints, bins, cumulative, density, range ? ImPlotRange(rmin,rmax) : ImPlotRange(), outliers);

				ImPlot::EndPlot();
			}

			ImGui::EndChild();
		}

		ImGui::End();
	}
}


