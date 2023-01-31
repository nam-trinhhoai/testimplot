/*
 *
 *
 *  Created on: 24 Aug 2022
 *      Author: l0359127
 */

#include "DragEllipse.h"

DragEllipse::DragEllipse(WorkingSetManager* manager):
m_manager(manager)
{}

DragEllipse::~DragEllipse()
{}


// Drag ellipse
void DragEllipse::plotDragEllipse(double &perimeterPoint_xMin, double &perimeterPoint_xMax, double &perimeterPoint_yMin, double &perimeterPoint_yMax, double log1Min, double log1Max, double log2Min, double log2Max)
{
	double centerPoint_x = (perimeterPoint_xMin + perimeterPoint_xMax)/2.0;
	double centerPoint_y = (perimeterPoint_yMin + perimeterPoint_yMax)/2.0;

	ImPlot::DragPoint(1,&perimeterPoint_xMin,&centerPoint_y,ImVec4(0,0,1,1),5,ImPlotDragToolFlags_None);
	ImPlot::DragPoint(2,&perimeterPoint_xMax,&centerPoint_y,ImVec4(0,0,1,1),5,ImPlotDragToolFlags_None);
	ImPlot::DragPoint(3,&centerPoint_x,&perimeterPoint_yMin,ImVec4(0,0,1,1),5,ImPlotDragToolFlags_None);
	ImPlot::DragPoint(4,&centerPoint_x,&perimeterPoint_yMax,ImVec4(0,0,1,1),5,ImPlotDragToolFlags_None);
	
	// Plot the ellipse
	static int const numEllipsePoints = 100;
	static ImPlotPoint ellipisePoints[numEllipsePoints];
	static double const PI = 3.14159265359;
	static double const dTheta = 2.0*PI/(numEllipsePoints-1);
	double ellipseHalfLength_x = (perimeterPoint_xMax - perimeterPoint_xMin)/2.0;	
	double ellipseHalfLength_y = (perimeterPoint_yMax - perimeterPoint_yMin)/2.0;
	for (int i=0; i<numEllipsePoints; i++)
	{
		double theta = i*dTheta;
		ellipisePoints[i] =ImPlotPoint(centerPoint_x+ellipseHalfLength_x*std::cos(theta), centerPoint_y+ellipseHalfLength_y*std::sin(theta));
	}
	ImPlot::SetNextLineStyle(ImVec4(0,0,1,1), 2);
	ImPlot::PlotLine("##DragEllipse",&ellipisePoints[0].x, &ellipisePoints[0].y, numEllipsePoints, 0, sizeof(ImPlotPoint));
}


// Cross-plot between the logs
void DragEllipse::showPlot() {
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
		static float log2Min = 1e6;
		static float log2Max = 1e-6;
		static float log1Min = 1e6;
		static float log1Max = 1e-6;

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
	
			// Plot styles
			//ImGui::ShowFontSelector("Font");
			ImGui::ShowStyleSelector("ImGui Style");
			ImPlot::ShowStyleSelector("ImPlot Style");
			ImPlot::ShowColormapSelector("ImPlot Colormap");
			ImGui::Checkbox("Anti-Aliased Lines", &ImPlot::GetStyle().AntiAliasedLines);
			
			// Plot
			if (ImPlot::BeginPlot((logName1+" vs "+logName2).c_str(), ImVec2(-1,-1))) 
			{
				ImPlot::SetupAxesLimits(log1Min, log1Max, log2Min, log2Max);
				ImPlot::SetupAxis(ImAxis_X1,logName1.c_str());
				ImPlot::SetupAxis(ImAxis_Y1,logName2.c_str());	
				
				if ((currentLog1.attributes.size()>0) & (currentLog2.attributes.size()>0))
				{
					ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(1,1,1,1), IMPLOT_AUTO, ImVec4(1,1,1,1));
					ImPlot::PlotScatter((logName1+" vs "+logName2).c_str(), log1, log2, numPoints);
				
					// Drag Points on the perimeter of the Ellipse
					static double perimeterPoint_xMin = (log1Min+log1Max)/2.0 - 0.5*(log1Max-log1Min)/2.0;// Starting radius is half of the maximum radius			
					static double perimeterPoint_xMax = (log1Min+log1Max)/2.0 + 0.5*(log1Max-log1Min)/2.0;
					static double perimeterPoint_yMin = (log2Min+log2Max)/2.0 - 0.5*(log2Max-log2Min)/2.0;
					static double perimeterPoint_yMax = (log2Min+log2Max)/2.0 + 0.5*(log2Max-log2Min)/2.0;

					plotDragEllipse(perimeterPoint_xMin, perimeterPoint_xMax, perimeterPoint_yMin, perimeterPoint_yMax, log1Min, log1Max, log2Min, log2Max);
					double ellipseHalfLength_x = (perimeterPoint_xMax-perimeterPoint_xMin)/2.0;
					double ellipseHalfLength_y = (perimeterPoint_yMax-perimeterPoint_yMin)/2.0;	
					double ellipseCenter_x = (perimeterPoint_xMax+perimeterPoint_xMin)/2.0;
					double ellipseCenter_y = (perimeterPoint_yMax+perimeterPoint_yMin)/2.0;
					
					// Highlight the selected points
					std::vector<double> log1Inner, log2Inner;
					for (int i=0; i<numPoints; i++)
					{	
						double val = (log1[i] - ellipseCenter_x)*(log1[i] - ellipseCenter_x)/ellipseHalfLength_x/ellipseHalfLength_x + (log2[i] - ellipseCenter_y)*(log2[i] - ellipseCenter_y)/ellipseHalfLength_y/ellipseHalfLength_y;
						if(val <= 1.0)
						{
							log1Inner.push_back(log1[i]);
							log2Inner.push_back(log2[i]);
						}
					}
					
					ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(1,0,0,1), IMPLOT_AUTO, ImVec4(1,0,0,1));
					ImPlot::PlotScatter("##InnerPoints", &log1Inner[0], &log2Inner[0], log1Inner.size());
				}


				ImPlot::EndPlot();
			}

			ImGui::EndChild();
		}

		ImGui::End();
	}
}

