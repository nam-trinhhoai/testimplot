/*
 *
 *
 *  Created on: 04 Aug 2022
 *      Author: l0359127
 */

#include "MultipleIntervalsCrossPlotWithHistogram.h"

MultipleIntervalsCrossPlotWithHistogram::MultipleIntervalsCrossPlotWithHistogram(WorkingSetManager* manager):
m_manager(manager)
{}

MultipleIntervalsCrossPlotWithHistogram::~MultipleIntervalsCrossPlotWithHistogram()
{}


// Cross-plot between the logs
void MultipleIntervalsCrossPlotWithHistogram::showPlot() {
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
		static ImGuiWindowFlags winflags = ImGuiWindowFlags_None;//ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;
		bool showPlot = true;
		
		static bool use_work_area = true;
		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImVec2 viewPortSize = viewport->WorkSize;

		ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);	
		ImGui::SetNextWindowSize(use_work_area ? ImVec2(viewPortSize.x*0.4, viewPortSize.y) : viewport->Size);
	
		ImGui::Begin("Logs", &showPlot, winflags);
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
		ImGui::BeginChild("ChildLeft", ImVec2(ImGui::GetContentRegionAvail().x*0.2f, -1), false, window_flags);

		// Variables defining the selected intervals
		static int numIntervals = 0;
		ImGui::SliderInt("Intervals", &numIntervals, 0, 3); // This is a design for maximum 3 intervals

		std::vector<double> selectedDepth[numIntervals], selectedLog1[numIntervals], selectedLog2[numIntervals];
	
		// Get data
		double *depth;
		double *log1, *log2;

		std::string logName1 = " ", logName2 = " ";

		// Initialization
		WellBore *bore;
		bool logIsSelected;
		Logs currentLog1, currentLog2;

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
							depth = &currentLog1.keys[0];
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
							//numPoints = currentLog2.attributes.size();

							log2 = &currentLog2.attributes[0];
							//depth = &currentLog2.keys[0];
						}
					}
				}
			}
		} // ImGui::TreeNode("Wellbores")

		ImGui::EndChild();

		ImGui::SameLine();

		// Plot
		//if(currentLog1.attributes.size() > 0)
		{
			static ImPlotRect lims(0,1,10000,0); // For linking depth axis of the two logs

			ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

			// Flag for the drag line
			static ImPlotDragToolFlags flags = ImPlotDragToolFlags_None;
	
			ImGui::BeginChild("ChildRight", ImVec2(-1, -1), true, window_flags);
	
			// Width of chart area for plotting the two logs
			double chartWidth = ImGui::GetContentRegionAvail().x;
	
			// Initialize the intervals top and bottom

			static double selectedTop[3] = {2000,2000,2000};  // Temporal solution
			static double selectedBottom[3] = {2100,2100,2100}; 
			static double anotationPos = 0;

			// Plot the first log
			if (ImPlot::BeginPlot("##Log1", ImVec2(0.57*chartWidth,-1), ImPlotFlags_NoLegend)) 
			{
				ImPlot::SetupAxis(ImAxis_X1,logName1.c_str());
				ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);	
				ImPlot::SetupAxisLinks(ImAxis_Y1, &lims.Y.Max, &lims.Y.Min); // For linking depth axis of the two logs
				if(currentLog1.attributes.size() > 0)
				{
					ImPlot::PlotLine(logName1.c_str(), log1, depth, numPoints);
										
					// Drag line for selecting interval top/bottom
					ImPlot::DragLineX(0,&anotationPos,ImVec4(1,1,1,1),1,flags);

					for (int interval_idx=0; interval_idx<numIntervals; interval_idx++)
					{
						ImPlot::DragLineY(4*interval_idx+1,&selectedTop[interval_idx],ImVec4(1,1,1,1),1,flags);
						ImPlot::DragLineY(4*interval_idx+2,&selectedBottom[interval_idx],ImVec4(1,1,1,1),1,flags);
						
						ImPlot::Annotation(anotationPos,selectedTop[interval_idx],ImVec4(1,1,1,1),ImVec2(0,0),false,("Top_"+std::to_string(interval_idx)).c_str());

						ImPlot::Annotation(anotationPos,selectedBottom[interval_idx],ImVec4(1,1,1,1),ImVec2(0,0),false,("Bottom_"+std::to_string(interval_idx)).c_str());

						// Highlight the selected interval
						selectedDepth[interval_idx].clear();
						selectedLog1[interval_idx].clear();
						selectedLog2[interval_idx].clear();
						for (int i=0; i<numPoints; i++)
						{
							if ((depth[i] >=selectedTop[interval_idx]) & (depth[i] <=selectedBottom[interval_idx]))
							{
								selectedDepth[interval_idx].push_back(depth[i]);
								selectedLog1[interval_idx].push_back(log1[i]);
								selectedLog2[interval_idx].push_back(log2[i]);
							}
						}

						ImPlot::SetNextLineStyle(ImVec4(interval_idx==0?1:0,interval_idx==1?1:0,interval_idx==2?1:0,1),1); // This is a design for maximum 3 intervals
						ImPlot::PlotLine("##SelectedInterval", &selectedLog1[interval_idx][0], &selectedDepth[interval_idx][0], selectedDepth[interval_idx].size());	

					}
				}				
				
				ImPlot::EndPlot();
			}

			// Plot the second log
			ImGui::SameLine();
			if (ImPlot::BeginPlot("##Log2", ImVec2(0.43*chartWidth,-1), ImPlotFlags_NoLegend)) // depth marker is removed from the second log, then it requires less area (40%) 
			{
				ImPlot::SetupAxis(ImAxis_X1,logName2.c_str());
				//ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);	
				ImPlot::SetupAxis(ImAxis_Y1,NULL, ImPlotAxisFlags_Invert | ImPlotAxisFlags_NoDecorations); // Depth marker of the second log is removed as it is linked to the depth of the first log.
				ImPlot::SetupAxisLinks(ImAxis_Y1, &lims.Y.Max, &lims.Y.Min); // For linking depth axis of the two logs

				if(currentLog2.attributes.size() > 0)
				{
					ImPlot::PlotLine(logName2.c_str(), log2, depth, numPoints);

					for (int interval_idx=0; interval_idx<numIntervals; interval_idx++)
					{
						ImPlot::DragLineY(4*interval_idx+3,&selectedTop[interval_idx],ImVec4(1,1,1,1),1,flags);
						ImPlot::DragLineY(4*interval_idx+4,&selectedBottom[interval_idx],ImVec4(1,1,1,1),1,flags);
						
						//ImPlot::Annotation(anotationPos,selectedTop[interval_idx],ImVec4(1,1,1,1),ImVec2(0,0),false,("Top_"+std::to_string(interval_idx)).c_str());

						//ImPlot::Annotation(anotationPos,selectedBottom[interval_idx],ImVec4(1,1,1,1),ImVec2(0,0),false,("Bottom_"+std::to_string(interval_idx)).c_str());

						// Highlight the selected interval
						selectedDepth[interval_idx].clear();
						selectedLog1[interval_idx].clear();
						selectedLog2[interval_idx].clear();
						for (int i=0; i<numPoints; i++)
						{
							if ((depth[i] >=selectedTop[interval_idx]) & (depth[i] <=selectedBottom[interval_idx]))
							{
								selectedDepth[interval_idx].push_back(depth[i]);
								selectedLog1[interval_idx].push_back(log1[i]);
								selectedLog2[interval_idx].push_back(log2[i]);
							}
						}

						ImPlot::SetNextLineStyle(ImVec4(interval_idx==0?1:0,interval_idx==1?1:0,interval_idx==2?1:0,1),1); // This is a design for maximum 3 intervals
						ImPlot::PlotLine("##Log2SelectedInterval", &selectedLog2[interval_idx][0], &selectedDepth[interval_idx][0], selectedDepth[interval_idx].size());	

					}
				}				
				
				ImPlot::EndPlot();
			}
			ImGui::EndChild();
		}

		ImGui::End();


		// Plot cross-plot and histograms of selected intervals
		ImGui::SetNextWindowPos(use_work_area ? ImVec2(viewport->WorkPos.x+viewPortSize.x*0.4, viewport->WorkPos.y) : viewport->Pos);	
		ImGui::SetNextWindowSize(use_work_area ? ImVec2(viewPortSize.x*0.6, viewPortSize.y) : viewport->Size);
			
		ImGui::Begin("Cross-Plot - Histograms", &showPlot, winflags);	
		// Plot
		{
			static bool cumulative = false;
			static bool density = false;
			static bool outliers = true;
			static int bins = 100;

			ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
	
			//ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
			ImGui::BeginChild("ChildRight", ImVec2(-1, -1), true, window_flags);

			// Place the three charts in a table for optimizing the view
			static ImGuiTableFlags table_flags = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
						                       ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable;

			if (ImGui::BeginTable("##table", 2, table_flags, ImVec2(-1,-1))) 
			{
				// Histogram of the first log
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);

				if (ImPlot::BeginPlot("##Histograms1", ImVec2(-1, 200), ImPlotFlags_NoLegend)) {
					if(currentLog1.attributes.size() > 0)
					{
						ImPlot::SetupAxis(ImAxis_X1,logName1.c_str());
						ImPlot::SetupAxis(ImAxis_Y1,"Count");
				
						for (int interval_idx=0; interval_idx<numIntervals; interval_idx++)
						{
							ImPlot::SetNextLineStyle(ImVec4(interval_idx==0?1:0,interval_idx==1?1:0,interval_idx==2?1:0,1),1); // This is a design for maximum 3 intervals
							ImPlot::PlotHistogramCurve(("Inteval_"+std::to_string(interval_idx)).c_str(), &selectedLog1[interval_idx][0], selectedLog1[interval_idx].size(), bins, cumulative, density, ImPlotRange(), outliers);
						}
					}
					ImPlot::EndPlot();
				}

				ImGui::TableNextRow();

				// Cross-plot
				ImGui::TableSetColumnIndex(0);

				if (ImPlot::BeginPlot("##CrossPlot", ImVec2(-1,-1))) 
				{
					if((currentLog1.attributes.size() > 0) & (currentLog2.attributes.size() > 0))
					{
						ImPlot::SetupAxis(ImAxis_X1,logName1.c_str());
						ImPlot::SetupAxis(ImAxis_Y1,logName2.c_str());
						
						for (int interval_idx=0; interval_idx<numIntervals; interval_idx++)
						{
							ImVec4 col = ImVec4(interval_idx==0?1:0,interval_idx==1?1:0,interval_idx==2?1:0,1); // This is designed for maximum 3 intervals
							ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, col, IMPLOT_AUTO, col);
							ImPlot::PlotScatter(("Inteval_"+std::to_string(interval_idx)).c_str(), &selectedLog1[interval_idx][0], &selectedLog2[interval_idx][0], selectedLog1[interval_idx].size());
						}
					}

					ImPlot::EndPlot();
				}

				// Histogram of the second log
				ImGui::TableSetColumnIndex(1);

				if (ImPlot::BeginPlot("##Histograms2", ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
					if(currentLog2.attributes.size() > 0)
					{
						ImPlot::SetupAxis(ImAxis_Y1,logName2.c_str());
						ImPlot::SetupAxis(ImAxis_X1,"Count");
				
						for (int interval_idx=0; interval_idx<numIntervals; interval_idx++)
						{
							ImPlot::SetNextLineStyle(ImVec4(interval_idx==0?1:0,interval_idx==1?1:0,interval_idx==2?1:0,1),1); // This is a design for maximum 3 intervals
							ImPlot::PlotHistogramHCurve(("Inteval_"+std::to_string(interval_idx)).c_str(), &selectedLog2[interval_idx][0], selectedLog2[interval_idx].size(), bins, cumulative, density, ImPlotRange(), outliers);
						}
					}
					ImPlot::EndPlot();
				}

				ImGui::EndTable();
			}

			ImGui::EndChild();
		}

		ImGui::End();
	}
}
