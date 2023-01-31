/*
 *
 *
 *  Created on: 10 Aug 2022
 *      Author: l0359127
 */

#include "CrossPlotRegression.h"

CrossPlotRegression::CrossPlotRegression(WorkingSetManager* manager):
m_manager(manager)
{}

CrossPlotRegression::~CrossPlotRegression()
{}

// Compute the linear regression coefficients
void CrossPlotRegression::LinearRegressionCoefficients(int numPoints, double *log1, double *log2, double &gradient, double &intersect)
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
void CrossPlotRegression::showPlot() {
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

			// Regression coefficients	
			static float gradientFactor = 1.0f;
			static float intersectFactor = 1.0f;
			
			ImGui::DragFloat("Gradient Multiplier", &gradientFactor, 0.001f, 0.9f, 1.1f, "%.03f");
			ImGui::DragFloat("Intersect Multiplier", &intersectFactor, 0.001f, 0.9f, 1.1f, "%.03f");

			static bool defaultFactors = false;
			ImGui::Checkbox("Reset Default Multipliers", &defaultFactors);
			if(defaultFactors)
			{
				gradientFactor = 1.0f;
				intersectFactor = 1.0f;
			}
		
			// Plot
			if (ImPlot::BeginPlot((logName1+" vs "+logName2).c_str(), ImVec2(-1,-1))) 
			{
				ImPlot::SetupAxesLimits(log1Min, log1Max, log2Min, log2Max);
				ImPlot::SetupAxis(ImAxis_X1,logName1.c_str());
				ImPlot::SetupAxis(ImAxis_Y1,logName2.c_str());	
				
				if ((currentLog1.attributes.size()>0) & (currentLog2.attributes.size()>0))
				{
					ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(0.5,0.5,0.5,0.5), IMPLOT_AUTO, ImVec4(0.5,0.5,0.5,0.5));
					ImPlot::PlotScatter((logName1+" vs "+logName2).c_str(), log1, log2, numPoints);
				
					// Plot linear regression
					double gradient, intersect;
					double log2Reg[numPoints];
					std::vector<double> log1Upper, log2Upper, log1Lower, log2Lower;

					LinearRegressionCoefficients(numPoints, log1, log2, gradient, intersect);
					gradient *= gradientFactor;
					intersect *= intersectFactor;
					
					for (int i=0; i<numPoints; i++)
					{
						log2Reg[i] = gradient*log1[i] + intersect;

						if(log2[i] > log2Reg[i])
						{
							log1Upper.push_back(log1[i]);
							log2Upper.push_back(log2[i]);
						}
						else
						{
							log1Lower.push_back(log1[i]);
							log2Lower.push_back(log2[i]);
						}
					}

					// Plot the regression line
					ImPlot::SetNextLineStyle(ImVec4(0,1,1,1),2);
					ImPlot::PlotLine("##Regression", log1, log2Reg, numPoints);

					// Initialize the position of the regression anotation at the position of the first point
					// Use a drag point allowings the anotation movable
					// Add regression function to the anotation
					static ImPlotPoint P = ImPlotPoint(log1[0],log2Reg[0]);
					ImPlot::DragPoint(0,&P.x,&P.y,ImVec4(1,1,1,1),2,ImPlotDragToolFlags_None);
					ImPlot::Annotation(P.x,P.y,ImVec4(1,1,1,1),ImVec2(15,-15),false,(logName2 + " = " + std::to_string(gradient) + " * " + logName1 + " + "+std::to_string(intersect)).c_str());
					
					// Highlight the points above the regression line
					ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(0,0,1,1), IMPLOT_AUTO, ImVec4(0,0,1,1));
					ImPlot::PlotScatter("##Upper", &log1Upper[0], &log2Upper[0], log1Upper.size());

					// Highlight the points below the regression line
					ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(0,1,0,1), IMPLOT_AUTO, ImVec4(0,1,0,1));
					ImPlot::PlotScatter("##Lower", &log1Lower[0], &log2Lower[0], log1Lower.size());
				}


				ImPlot::EndPlot();
			}

			ImGui::EndChild();
		}

		ImGui::End();
	}
}

