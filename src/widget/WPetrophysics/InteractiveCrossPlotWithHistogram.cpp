/*
 *
 *
 *  Created on: 01 Aug 2022
 *      Author: l0359127
 */

#include "InteractiveCrossPlotWithHistogram.h"

InteractiveCrossPlotWithHistogram::InteractiveCrossPlotWithHistogram(WorkingSetManager* manager):
m_manager(manager)
{}

InteractiveCrossPlotWithHistogram::~InteractiveCrossPlotWithHistogram()
{}

// Interactive helper			
int InteractiveCrossPlotWithHistogram::interactiveHelper(double *log1, double *log2, int numPoints)
{
	double log1Min=1e6, log1Max=1e-6, log2Min=1e6, log2Max=1e-6;
	double nullValue = -999.25;
	
	for( int i=0; i<numPoints; i++)
	{
		if (log1[i] != nullValue)
		{
			if(log1[i] > log1Max)
				log1Max = log1[i];
			if(log1[i] < log1Min)
				log1Min = log1[i];
		}

		if (log2[i] != nullValue)
		{
			if(log2[i] > log2Max)
				log2Max = log2[i];
			if(log2[i] < log2Min)
				log2Min = log2[i];
		}
	}	

	double log1Range = log1Max - log1Min;
	double log2Range = log2Max - log2Min;

	// Identify the index of the current point
	int point_idx = -1;
	double tolerance = 1e6;
	double distance = 1e6;
 
	ImPlotPoint plotMousePos = ImPlot::GetPlotMousePos(IMPLOT_AUTO,IMPLOT_AUTO);

	for (int i=0; i<numPoints; i++)
	{
		if (log1Range*log2Range != 0)
			distance = std::sqrt((log1[i] - plotMousePos.x)*(log1[i] - plotMousePos.x)/log1Range/log1Range + (log2[i] - plotMousePos.y)*(log2[i] - plotMousePos.y)/log2Range/log2Range);			
		
		if (distance < tolerance)
		{
			point_idx = i;
			tolerance = distance;		
		}	
	}

	// Show the values at mouse position
	ImGui::SetTooltip("point_idx=%d, x=%2f, y=%2f", point_idx, plotMousePos.x, plotMousePos.y);

	return point_idx;
}


// Interactive histogram helper			
int InteractiveCrossPlotWithHistogram::interactiveHistogramHelper(int point_idx, double *log, int numPoints, int bins)
{
	double logMin = 1e6;
	double logMax = 1e-6;
	double nullValue = -999.25;

	for (int i=0; i<numPoints; i++)
	{
		if (log[i] > nullValue)
		{
			if (log[i] > logMax)
				logMax = log[i];
			if (log[i] < logMin)
				logMin = log[i];	
		}	
	}

	double xBins[bins];
	double deltaXBin = (logMax-logMin)/bins;

	for (int i=0; i<bins; i++)
	{
		xBins[i] = logMin + (i+0.5)*deltaXBin;
	}

	//ImPlotPoint plotMousePos = ImPlot::GetPlotMousePos(IMPLOT_AUTO,IMPLOT_AUTO);
	
	// Identify the index of the current bin

	int bin_idx = -1;

	double tolerance = 1e6;

	for (int i=0; i<bins; i++)
	{
		double distance = std::abs(xBins[i] - log[point_idx]);
		
		if (distance < tolerance)
		{
			bin_idx = i;
			tolerance = distance;		
		}	
	}

	// Count points within active bin
	double xActiveBinMin = xBins[bin_idx] - 0.5*deltaXBin;
	double xActiveBinMax = xBins[bin_idx] + 0.5*deltaXBin;

	int countPointsInActiveBin = 0;
	std::vector<double> activePoints;
	for (int i=0; i<numPoints; i++)
	{
		if (log[i] > nullValue)
		{			
			if ((log[i] >= xActiveBinMin) & (log[i] < xActiveBinMax))
			{
				countPointsInActiveBin += 1;
				activePoints.push_back(log[i]);
			}
		}	
	}
	
	// If the active bin is the the last bin, add the last point to it
	if (bin_idx == bins-1)
	{
		countPointsInActiveBin += 1;
		activePoints.push_back(log[numPoints-1]);
	} 

	// Show the values at mouse position
	//ImGui::SetTooltip("bin_idx=%d, count=%d, xMin=%2f, xMax=%2f", bin_idx, countPointsInActiveBin, xActiveBinMin, xActiveBinMax);

	return bin_idx;
}

// Cross-plot between the logs
void InteractiveCrossPlotWithHistogram::showPlot() {
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

				// List of logs fir selecting the first log
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


				// List of logs fir selecting the second log
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

			// Place the three charts in a table for optimizing the view
			static ImGuiTableFlags table_flags = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
						                       ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable;

			if (ImGui::BeginTable("##table", 2, table_flags, ImVec2(-1,-1))) 
			{
				// Histogram of the first log
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);

				if (ImPlot::BeginPlot("##Histograms1", ImVec2(-1, 200))) {
					ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);
					ImPlot::SetupAxis(ImAxis_Y1,"Count");	

					if (currentLog1.attributes.size()>0)
					{
						ImPlot::PlotHistogram(logName1.c_str(), log1, numPoints, bins, cumulative, density, ImPlotRange(), outliers);

						// Highlight the corresponding active bin
						bin1_idx = interactiveHistogramHelper(point_idx, log1, numPoints, bins);
						if (bin1_idx>0)
						{
							ImPlot::PlotHistogramHighlight(bin1_idx, logName1.c_str(), log1, numPoints, bins, cumulative, density, ImPlotRange(), outliers);
						}
					}

					ImPlot::EndPlot();
				}

				ImGui::TableNextRow();

				// Cross-plot
				ImGui::TableSetColumnIndex(0);

				if (ImPlot::BeginPlot("##CrossPlot", ImVec2(-1,-1))) 
				{
					ImPlot::SetupAxesLimits(log1Min, log1Max, log2Min, log2Max);
					ImPlot::SetupAxis(ImAxis_X1,logName1.c_str());
					ImPlot::SetupAxis(ImAxis_Y1,logName2.c_str());	
				
					if ((currentLog1.attributes.size()>0) & (currentLog2.attributes.size()>0))
					{
						ImPlot::PlotScatter((logName1+" vs "+logName2).c_str(), log1, log2, numPoints);
						point_idx = interactiveHelper(log1, log2, numPoints);

						ImPlot::PlotScatterHighlight(point_idx, (logName1+" vs "+logName2).c_str(), log1, log2, numPoints);	
					}

					ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
					ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 6, ImPlot::GetColormapColor(1), IMPLOT_AUTO, ImPlot::GetColormapColor(1));
					ImPlot::PopStyleVar();

					ImPlot::EndPlot();
				}

				// Histogram of the second log
				ImGui::TableSetColumnIndex(1);

				if (ImPlot::BeginPlot("##Histograms2", ImVec2(-1, -1))) {
					ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);
					ImPlot::SetupAxis(ImAxis_X1,"Count");	

					if (currentLog2.attributes.size()>0)
					{
						ImPlot::PlotHistogramH(logName2.c_str(), log2, numPoints, bins, cumulative, density, ImPlotRange(), outliers);

						// Highlight the corresponding active bin
						bin2_idx = interactiveHistogramHelper(point_idx, log2, numPoints, bins);
						if (bin2_idx>0)
						{
							ImPlot::PlotHistogramHHighlight(bin2_idx, logName2.c_str(), log2, numPoints, bins, cumulative, density, ImPlotRange(), outliers);
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

