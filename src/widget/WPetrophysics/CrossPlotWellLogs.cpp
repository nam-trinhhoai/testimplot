/*
 *
 *
 *  Created on: 18 Jul 2022
 *      Author: l0359127
 */

#include "CrossPlotWellLogs.h"

CrossPlotWellLogs::CrossPlotWellLogs(WorkingSetManager* manager):
m_manager(manager)
{}

CrossPlotWellLogs::~CrossPlotWellLogs()
{}


// Cross-plot between the logs
void CrossPlotWellLogs::showPlot() {
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
		double *log2;

		std::string logName1 = " ";
		std::string logName2 = " ";
		float const nullValue = -999.25;
		float log2Min = 1e6;
		float log2Max = 1e-6;
		float log1Min = 1e6;
		float log1Max = 1e-6;

		// Initialization
		WellBore *bore;
		bool logIsSelected;
		Logs currentLog1;
		Logs currentLog2;

		if (ImGui::TreeNode("WellBores"))
		{
			// Select a well
			static int selectedWell = -1;
			
			//for (int n = 0; n < wellHead->wellBores().size(); n++)
			for (int n = 0; n < totalNumberOfWellBores; n++)
			{
				//if (ImGui::Selectable(wellHead->wellBores()[n]->name().toStdString().c_str(), selectedWell == n))
				if (ImGui::Selectable(listWellBores[n]->name().toStdString().c_str(), selectedWell == n))
					selectedWell = n;
			}
			ImGui::TreePop();

			if(selectedWell >-1)
			{
				//bore = wellHead->wellBores()[selectedWell];
				bore = listWellBores[selectedWell];
	

				// Get log data from selected well
				std::vector<std::string> logNames;
				for (int i=0; i<bore->logsNames().size();i++)
				{
					logNames.push_back( bore->logsNames()[i].toStdString() );	
				}

				int numLogs = bore->logsNames().size();

				// List of logs for selecting the first log
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

							for(int i=0; i<numPoints; i++)
							{
								// Update data range
								if(log1[i] != nullValue)
								{
									if(log1[i] < log1Min)
										log1Min = log1[i];
									if(log1[i] > log1Max)
										log1Max = log1[i];
								}
							}
						}
					}
				}


				// List of logs for selecting the second log
				{
					if (ImGui::TreeNode("Select the second log"))
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
							logName2 = bore->logsNames()[selected].toStdString();

							logIsSelected = bore->selectLog(selected);

							currentLog2 = bore->currentLog();
							//numPoints = currentLog.attributes.size();

							log2 = &currentLog2.attributes[0];

							for(int i=0; i<numPoints; i++)
							{
								// Update data range
								if(log2[i] != nullValue)
								{
									if(log2[i] < log2Min)
										log2Min = log2[i];
									if(log2[i] > log2Max)
										log2Max = log2[i];
								}
							}
						}
					}
				}
			}

		} // ImGui::TreeNode("Wells")

		ImGui::EndChild();

		ImGui::SameLine();

		// Plot
		{
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
	
			//ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
			ImGui::BeginChild("ChildRight", ImVec2(-1, -1), true, window_flags);
			if (ImPlot::BeginPlot((logName1+" vs "+logName2).c_str(), ImVec2(-1,-1))) 
			{
				ImPlot::SetupAxesLimits(log1Min, log1Max, log2Min, log2Max);
				ImPlot::SetupAxis(ImAxis_X1,logName1.c_str());
				ImPlot::SetupAxis(ImAxis_Y1,logName2.c_str());	
				
				if ((currentLog1.attributes.size()>0) & (currentLog2.attributes.size()>0))
					ImPlot::PlotScatter((logName1+" vs "+logName2).c_str(), log1, log2, numPoints);

				ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
				ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 6, ImPlot::GetColormapColor(1), IMPLOT_AUTO, ImPlot::GetColormapColor(1));
				ImPlot::PopStyleVar();

				ImPlot::EndPlot();
			}

			ImGui::EndChild();
		}

		ImGui::End();
	}
}

