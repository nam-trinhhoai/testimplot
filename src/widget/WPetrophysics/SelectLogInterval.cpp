/*
 *
 *
 *  Created on: 01 Aug 2022
 *      Author: l0359127
 */

#include "SelectLogInterval.h"

SelectLogInterval::SelectLogInterval(WorkingSetManager* manager):
m_manager(manager)
{}

SelectLogInterval::~SelectLogInterval()
{}

// Plot the well logs from database in quickplot format
void SelectLogInterval::showPlot() {
	// Selected data
	std::vector<double> selectedDepth, selectedLog;

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

	int totalNumberOfWellBores = listWellBores.size();
	if(totalNumberOfWellBores > 0)
	{
		static ImGuiWindowFlags winflags = ImGuiWindowFlags_None;//ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;
		bool showPlot = true;
		
		static bool use_work_area = true;
		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImVec2 viewPortSize = viewport->WorkSize;

		ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);	
		ImGui::SetNextWindowSize(use_work_area ? ImVec2(viewPortSize.x*0.5, viewPortSize.y) : viewport->Size);
	
		ImGui::Begin("Plot a single log", &showPlot, winflags);
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
		ImGui::BeginChild("ChildLeft", ImVec2(ImGui::GetContentRegionAvail().x*0.2f, -1), false, window_flags);

		// Get data
		double *depth;
		double *log;

		std::string logName = " ";

		// Initialization
		WellBore *bore;
		bool logIsSelected;
		Logs currentLog;

		if (ImGui::TreeNode("WellBores"))
		{
			// Select a well
			static int selectedWell = -1;
			
			for (int n = 0; n < totalNumberOfWellBores; n++)
			{
				if (ImGui::Selectable(listWellBores[n]->name().toStdString().c_str(), selectedWell == n))
					selectedWell = n;
			}
			ImGui::TreePop();

			if(selectedWell >-1)
			{
				bore = listWellBores[selectedWell];
				int numLogs = bore->logsNames().size();
	

				// Get log data from selected well
				std::vector<std::string> logNames;
				for (int i=0; i<bore->logsNames().size();i++)
				{
					logNames.push_back( bore->logsNames()[i].toStdString() );	
				}

				// List of logs
				{
					if (ImGui::TreeNode("Logs"))
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
							logName = bore->logsNames()[selected].toStdString();
							logIsSelected = bore->selectLog(selected);

							currentLog = bore->currentLog();
							numPoints = currentLog.attributes.size();

							log = &currentLog.attributes[0];
							depth = &currentLog.keys[0];
						}
					}
				}
			}

		} // ImGui::TreeNode("Wellbores")

		ImGui::EndChild();

		ImGui::SameLine();

		// Plot
		{
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

			// Flag for the drag line
			static ImPlotDragToolFlags flags = ImPlotDragToolFlags_None;
	
			ImGui::BeginChild("ChildRight", ImVec2(-1, -1), true, window_flags);
			if (ImPlot::BeginPlot(logName.c_str(), ImVec2(-1,-1))) 
			{
				ImPlot::SetupAxis(ImAxis_X1,logName.c_str());
				ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);	
				if(currentLog.attributes.size() > 0)
				{
					ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);
										
					// Drag line for selecting interval top/bottom
					double depthMin = 1e6;
					double depthMax = 1e-6;

					for (int i=0; i<numPoints; i++)
					{
						if (depth[i] < depthMin)
							depthMin = depth[i];
						
						if (depth[i] > depthMax)
							depthMax = depth[i];
					}

					static double selectedTop = depthMin;
					static double selectedBottom = depthMax;
					static double anotationPos = 0;

					ImPlot::DragLineX(0,&anotationPos,ImVec4(1,1,1,1),1,flags);

					ImPlot::DragLineY(1,&selectedTop,ImVec4(1,0,0,1),1,flags);
					ImPlot::Annotation(anotationPos,selectedTop,ImVec4(1,0,0,1),ImVec2(0,0),false,("Top ("+std::to_string(selectedTop)+")").c_str());

					ImPlot::DragLineY(2,&selectedBottom,ImVec4(0,1,0,1),1,flags);
					ImPlot::Annotation(anotationPos,selectedBottom,ImVec4(0,1,0,1),ImVec2(0,0),false,("Bottom ("+std::to_string(selectedBottom)+")").c_str());

					// Highlight the selected interval
					selectedDepth.clear();
					selectedLog.clear();
					for (int i=0; i<numPoints; i++)
					{
						if ((depth[i] >=selectedTop) & (depth[i] <=selectedBottom))
						{
							selectedDepth.push_back(depth[i]);
							selectedLog.push_back(log[i]);
						}
					}

					ImPlot::SetNextLineStyle(ImVec4(1,1,1,1),1);
					ImPlot::PlotLine("##SelectedInterval", &selectedLog[0], &selectedDepth[0], selectedDepth.size());	
				}				
				
				ImPlot::EndPlot();
			}
			ImGui::EndChild();
		}

		ImGui::End();

		// Plot histogram of selected interval
		ImGui::SetNextWindowPos(use_work_area ? ImVec2(viewport->WorkPos.x+viewPortSize.x*0.5, viewport->WorkPos.y) : viewport->Pos);	
		ImGui::SetNextWindowSize(use_work_area ? ImVec2(viewPortSize.x*0.5, viewPortSize.y) : viewport->Size);
				
		ImGui::Begin("Histogram", &showPlot, winflags);
		static bool cumulative = false;
		static bool density = false;
		static bool outliers = true;
		static int bins = 100;

		if (ImPlot::BeginPlot("##Histogram", ImVec2(-1,-1))) 
		{
			if(currentLog.attributes.size() > 0)
			{
				ImPlot::SetupAxis(ImAxis_X1,logName.c_str());
				ImPlot::SetupAxis(ImAxis_Y1,"Count");
				ImPlot::SetNextLineStyle(ImVec4(1,1,1,1),1);
				ImPlot::PlotHistogram("##SelectedIntervalHistogram", &selectedLog[0], selectedLog.size(), bins, cumulative, density, ImPlotRange(), outliers);
			}
			ImPlot::EndPlot();
		}
		ImGui::End();


	}
}
