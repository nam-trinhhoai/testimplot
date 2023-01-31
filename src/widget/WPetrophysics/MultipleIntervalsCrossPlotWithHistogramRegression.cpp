/*
 *
 *
 *  Created on: 11 Aug 2022
 *      Author: l0359127
 */

#include "MultipleIntervalsCrossPlotWithHistogramRegression.h"

MultipleIntervalsCrossPlotWithHistogramRegression::MultipleIntervalsCrossPlotWithHistogramRegression(WorkingSetManager* manager):
m_manager(manager)
{}

MultipleIntervalsCrossPlotWithHistogramRegression::~MultipleIntervalsCrossPlotWithHistogramRegression()
{}

// Compute the linear regression coefficients
void MultipleIntervalsCrossPlotWithHistogramRegression::LinearRegressionCoefficients(int numPoints, double *log1, double *log2, double &gradient, double &intersect)
{
	double sumX = 0;
	double sumX2 = 0;
	double sumY = 0;
	double sumXY = 0;
	
	for (int i=0; i<numPoints; i++)
	{
		sumX += log1[i];
		sumX2 += log1[i]*log1[i];
		sumY += log2[i];
		sumXY += log1[i]*log2[i];
	}
	
	double denom = numPoints * sumX2 - sumX*sumX;
	if(denom != 0)
	{
		gradient = (sumXY*numPoints - sumX*sumY) / denom;
		intersect = (sumX2*sumY - sumXY*sumX) / denom;
	}
	else
	{
		gradient = 1;
		intersect = 0;
	}
}

// Cross-plot between the logs
void MultipleIntervalsCrossPlotWithHistogramRegression::showPlot() {
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

		// Select an interval for regression
		static int regInterval = 0;
		ImGui::SliderInt("Reg Interval", &regInterval, 0, numIntervals); // regInterval=0 show data of all intervals without regression

		// Add a plot showing distances from each cross-plot point to the regression line
		int numLogCharts;
		if (regInterval > 0)
			numLogCharts = 3;
		else
			numLogCharts = 2;

		// Vectors storing data of selected intervals
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
			if (ImPlot::BeginPlot("##Log1", ImVec2((0.07+1.0/numLogCharts)*chartWidth,-1), ImPlotFlags_NoLegend)) 
			{
				ImPlot::SetupAxis(ImAxis_X1,logName1.c_str());
				ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);	
				ImPlot::SetupAxisLinks(ImAxis_Y1, &lims.Y.Max, &lims.Y.Min); // For linking depth axis of the two logs
				if((currentLog1.attributes.size() > 0) & (currentLog2.attributes.size() > 0))
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
			if (ImPlot::BeginPlot("##Log2", ImVec2((1.0/numLogCharts-0.07/(numLogCharts-1))*chartWidth,-1), ImPlotFlags_NoLegend)) // depth marker is removed from the second log, then it requires less area (40%) 
			{
				ImPlot::SetupAxis(ImAxis_X1,logName2.c_str());
				//ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);	
				ImPlot::SetupAxis(ImAxis_Y1,NULL, ImPlotAxisFlags_Invert | ImPlotAxisFlags_NoDecorations); // Depth marker of the second log is removed as it is linked to the depth of the first log.
				ImPlot::SetupAxisLinks(ImAxis_Y1, &lims.Y.Max, &lims.Y.Min); // For linking depth axis of the two logs

				if((currentLog1.attributes.size() > 0) & (currentLog2.attributes.size() > 0))
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

			// Add third log showing the distance from cross-plot points to the regression line
			if(numLogCharts == 3)
			{
				ImGui::SameLine();
				if (ImPlot::BeginPlot("##Log3", ImVec2((1.0/numLogCharts-0.07/(numLogCharts-1))*chartWidth,-1))) 
				{
					ImPlot::SetupAxis(ImAxis_X1,"Distance");
					ImPlot::SetupAxis(ImAxis_Y1,NULL, ImPlotAxisFlags_Invert | ImPlotAxisFlags_NoDecorations);
					ImPlot::SetupAxisLinks(ImAxis_Y1, &lims.Y.Max, &lims.Y.Min); // For linking depth axis of the two logs

					if((currentLog1.attributes.size() > 0) & (currentLog2.attributes.size() > 0))
					{
						ImU32 upperColor = ImGui::ColorConvertFloat4ToU32(ImVec4(0, 0, 1, 1));
						ImU32 lowerColor = ImGui::ColorConvertFloat4ToU32(ImVec4(0, 1, 0, 1));

						for (int interval_idx=0; interval_idx<numIntervals; interval_idx++)
						{
							if(interval_idx+1 == regInterval)
							{
								// Compute linear regression line
								int sizeSelectedInterval = selectedLog1[interval_idx].size();
								double gradient, intersect;
								double log2Reg[sizeSelectedInterval];
								std::vector<double> distanceUpper, distanceLower, threshold;

								LinearRegressionCoefficients(sizeSelectedInterval, &selectedLog1[interval_idx][0], &selectedLog2[interval_idx][0], gradient, intersect);
					
								for (int i=0; i<sizeSelectedInterval; i++)
								{
									log2Reg[i] = gradient*selectedLog1[interval_idx][i] + intersect;
								
									if(selectedLog2[interval_idx][i] > log2Reg[i])
									{
										distanceUpper.push_back(selectedLog2[interval_idx][i] - log2Reg[i]);
										distanceLower.push_back(0);
									}
									else
									{
										distanceUpper.push_back(0);
										distanceLower.push_back(selectedLog2[interval_idx][i]-log2Reg[i]);
									}
									threshold.push_back(0);
								}

								// Show the distances from the cross-plot points to the regression line
								if(sizeSelectedInterval > 0)
								{
									ImPlot::PlotShadedV(("##Distance_Upper_Shaded_"+std::to_string(interval_idx)).c_str(), &distanceUpper[0], &threshold[0], &selectedDepth[interval_idx][0], sizeSelectedInterval, upperColor);
									ImPlot::PlotShadedV(("##Distance_Lower_Shaded_"+std::to_string(interval_idx)).c_str(), &distanceLower[0], &threshold[0], &selectedDepth[interval_idx][0], sizeSelectedInterval, lowerColor);
								}
							}
						}	
					}		

					ImPlot::EndPlot();
				}
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

							if((regInterval == 0) | (interval_idx+1 == regInterval)) // regInterval=0 show data of all intervals without regression
							{
								ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, col, IMPLOT_AUTO, col);
								ImPlot::PlotScatter(("Inteval_"+std::to_string(interval_idx)).c_str(), &selectedLog1[interval_idx][0], &selectedLog2[interval_idx][0], selectedLog1[interval_idx].size());
							}

							if(interval_idx+1 == regInterval)
							{
								// Compute linear regression line
								int sizeSelectedInterval = selectedLog1[interval_idx].size();
								double gradient, intersect;
								double log2Reg[sizeSelectedInterval];
								std::vector<double> log1Upper, log2Upper, log1Lower, log2Lower;

								LinearRegressionCoefficients(sizeSelectedInterval, &selectedLog1[interval_idx][0], &selectedLog2[interval_idx][0], gradient, intersect);
				
								for (int i=0; i<sizeSelectedInterval; i++)
								{
									log2Reg[i] = gradient*selectedLog1[interval_idx][i] + intersect;

									if(selectedLog2[interval_idx][i] > log2Reg[i])
									{
										log1Upper.push_back(selectedLog1[interval_idx][i]);
										log2Upper.push_back(selectedLog2[interval_idx][i]);
									}
									else
									{
										log1Lower.push_back(selectedLog1[interval_idx][i]);
										log2Lower.push_back(selectedLog2[interval_idx][i]);
									}
								}

								// Plot the regression line
								if(sizeSelectedInterval>0)
								{
									ImPlot::SetNextLineStyle(col,2);
									ImPlot::PlotLine(("##Regression_"+std::to_string(interval_idx)).c_str(), &selectedLog1[interval_idx][0], log2Reg, sizeSelectedInterval);
				
									// Highlight the points above the regression line
									ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(0,0,1,1), IMPLOT_AUTO, ImVec4(0,0,1,1));
									ImPlot::PlotScatter(("##Upper_"+std::to_string(interval_idx)).c_str(), &log1Upper[0], &log2Upper[0], log1Upper.size());

									// Highlight the points below the regression line
									ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(0,1,0,1), IMPLOT_AUTO, ImVec4(0,1,0,1));
									ImPlot::PlotScatter(("##Lower_"+std::to_string(interval_idx)).c_str(), &log1Lower[0], &log2Lower[0], log1Lower.size());
								}
							}
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

