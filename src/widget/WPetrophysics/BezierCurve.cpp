/*
 *
 *
 *  Created on: 29 Aug 2022
 *      Author: l0359127
 */

#include "BezierCurve.h"

BezierCurve::BezierCurve(WorkingSetManager* manager):
m_manager(manager)
{}

BezierCurve::~BezierCurve()
{}

// Drag Bezier curve
void BezierCurve::plotBezierCurve(double &perimeterPoint_xMin, double &perimeterPoint_xMax, double &perimeterPoint_yMin, double &perimeterPoint_yMax, double log1Min, double log1Max, double log2Min, double log2Max)
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
	ImPlot::PlotLine("##BezierCurve",&ellipisePoints[0].x, &ellipisePoints[0].y, numEllipsePoints, 0, sizeof(ImPlotPoint));
}


// Cross-plot between the logs
void BezierCurve::showPlot() {
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
					//ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(1,1,1,1), IMPLOT_AUTO, ImVec4(1,1,1,1));
					//ImPlot::PlotScatter((logName1+" vs "+logName2).c_str(), log1, log2, numPoints);
		
					// Estimate the initial positions of the drag points regarding the size of the cross-plot area.
					static double perimeterPoint_xMin = (log1Min+log1Max)/2.0 - 0.5*(log1Max-log1Min)/2.0;// Starting radius is half of the maximum radius			
					static double perimeterPoint_xMax = (log1Min+log1Max)/2.0 + 0.5*(log1Max-log1Min)/2.0;
					static double perimeterPoint_yMin = (log2Min+log2Max)/2.0 - 0.5*(log2Max-log2Min)/2.0;
					static double perimeterPoint_yMax = (log2Min+log2Max)/2.0 + 0.5*(log2Max-log2Min)/2.0;
					double centerPoint_x = (perimeterPoint_xMin + perimeterPoint_xMax)/2.0;
					double centerPoint_y = (perimeterPoint_yMin + perimeterPoint_yMax)/2.0;

					static ImPlotPoint P[] = {ImPlotPoint(perimeterPoint_xMin,centerPoint_y), 
											  ImPlotPoint(centerPoint_x,perimeterPoint_yMin),  
											  ImPlotPoint(centerPoint_x,perimeterPoint_yMin),  
											  ImPlotPoint(perimeterPoint_xMax,centerPoint_y)};

					// Plot the drag points
					static ImPlotDragToolFlags dragPointFrags = ImPlotDragToolFlags_None;	
					
					ImPlot::DragPoint(0,&P[0].x,&P[0].y, ImVec4(0,0.9f,0,1),4,dragPointFrags);
					ImPlot::DragPoint(1,&P[1].x,&P[1].y, ImVec4(1,0.5f,1,1),4,dragPointFrags);
					ImPlot::DragPoint(2,&P[2].x,&P[2].y, ImVec4(0,0.5f,1,1),4,dragPointFrags);
					ImPlot::DragPoint(3,&P[3].x,&P[3].y, ImVec4(0,0.9f,0,1),4,dragPointFrags);
		
					int pt_idx = 3;
					if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0) && ImGui::GetIO().KeyCtrl) {
						ImPlotPoint pt = ImPlot::GetPlotMousePos();
						static double pt_x = pt.x;	
						static double pt_y = pt.y;
						pt_idx += 1;
						ImPlot::DragPoint(pt_idx,&pt_x,&pt_y, ImVec4(0,0.9f,0,1),4,dragPointFrags);
					}


					// Plot the Bezier curve	
					static ImPlotPoint B[200];
					for (int i = 0; i < 100; ++i) {
						double t  = i / 99.0;
						double u  = 1 - t;
						double w1 = u*u*u;
						double w2 = 3*u*u*t;
						double w3 = 3*u*t*t;
						double w4 = t*t*t;
						B[i] = ImPlotPoint(w1*P[0].x + w2*P[1].x + w3*P[2].x + w4*P[3].x, w1*P[0].y + w2*P[1].y + w3*P[2].y + w4*P[3].y);
						B[199-i] = ImPlotPoint(w1*P[0].x + w2*(2.0*P[0].x-P[1].x) + w3*(2.0*P[3].x-P[2].x) + w4*P[3].x, w1*P[0].y + w2*(2.0*P[0].y-P[1].y) + w3*(2.0*P[3].y-P[2].y) + w4*P[3].y);
					}


					ImPlot::SetNextLineStyle(ImVec4(1,0.5f,1,1));
					ImPlot::PlotLine("##h1",&P[0].x, &P[0].y, 2, 0, sizeof(ImPlotPoint));
					ImPlot::SetNextLineStyle(ImVec4(0,0.5f,1,1));
					ImPlot::PlotLine("##h2",&P[2].x, &P[2].y, 2, 0, sizeof(ImPlotPoint));
					ImPlot::SetNextLineStyle(ImVec4(0,0.9f,0,1), 2);
					ImPlot::PlotLine("##bez",&B[0].x, &B[0].y, 200, 0, sizeof(ImPlotPoint));

					// Highlight the selected points

						static double val_x=0;
						static double val_y=0;
						ImPlot::DragPoint(4,&val_x,&val_y, ImVec4(1,0,0,1),6,dragPointFrags);;

						bool isInner = false;
						for (int i_bezier = 0; i_bezier < 10; ++i_bezier) 
						{
							double t  = i_bezier / 9.0;
							double u  = 1 - t;
							double w1 = u*u*u;
							double w2 = 3*u*u*t;
							double w3 = 3*u*t*t;
							double w4 = t*t*t;
							double val_x_1 = w1*P[0].x + w2*P[1].x + w3*P[2].x + w4*P[3].x;
							double val_y_1 = w1*P[0].y + w2*P[1].y + w3*P[2].y + w4*P[3].y;
							double val_dx_dt_1 = -3.0*u*u*P[0].x + (-6.0*u*t+3*u*u)*P[1].x + (-3.0*t*t+6.0*u*t)*P[2].x + 3.0*t*t*P[3].x;
							double val_dy_dt_1 = -3.0*u*u*P[0].y + (-6.0*u*t+3*u*u)*P[1].y  + (-3.0*t*t+6.0*u*t)*P[2].y + 3.0*t*t*P[3].y;
							double val_1 = (val_y - val_y_1) - val_dy_dt_1/val_dx_dt_1*(val_x - val_x_1);
							std::cout << "val_1 is positive " << std::to_string(val_1) << std::endl;
						}

						for (int i_bezier = 0; i_bezier < 10; ++i_bezier) 
						{
							double t  = i_bezier / 9.0;
							double u  = 1 - t;
							double w1 = u*u*u;
							double w2 = 3*u*u*t;
							double w3 = 3*u*t*t;
							double w4 = t*t*t;

							double val_x_2 = w1*P[0].x + w2*(2.0*P[0].x-P[1].x) + w3*(2.0*P[3].x-P[2].x) + w4*P[3].x;
							double val_y_2 = w1*P[0].y + w2*(2.0*P[0].y-P[1].y) + w3*(2.0*P[3].y-P[2].y) + w4*P[3].y;
							double val_dx_dt_2 = -3.0*u*u*P[0].x + (-6.0*u*t+3*u*u)*(2.0*P[0].x-P[1].x) + (-3.0*t*t+6.0*u*t)*(2.0*P[3].x-P[2].x) + 3.0*t*t*P[3].x;
							double val_dy_dt_2 = -3.0*u*u*P[0].y + (-6.0*u*t+3*u*u)*(2.0*P[0].y-P[1].y) + (-3.0*t*t+6.0*u*t)*(2.0*P[3].y-P[2].y) + 3.0*t*t*P[3].y;
							double val_2 = (val_y - val_y_2) - val_dy_dt_2/val_dx_dt_2*(val_x - val_x_2);
							std::cout << "val_2 is positive " << std::to_string(val_2) << std::endl;
						}


/**
					std::vector<double> log1Inner, log2Inner;
					for (int i=0; i<numPoints; i++)
					{	
						
						
						{
							log1Inner.push_back(log1[i]);
							log2Inner.push_back(log2[i]);
						}
					}
					
					ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(1,0,0,1), IMPLOT_AUTO, ImVec4(1,0,0,1));
					ImPlot::PlotScatter("##InnerPoints", &log1Inner[0], &log2Inner[0], log1Inner.size());	
*/
/**			
					// Highlight the selected points
					std::vector<double> log1Inner, log2Inner;
					for (int i=0; i<numPoints; i++)
					{	
						// Search for the intersects between the vertical line passing through a given point and the Bezier curves
						double tolerance_1 = 1e6;
						double tolerance_2 = 1e6;
						int i_bezier_1 = 0;
						int i_bezier_2 = 0;
						int bezier_curve_idx_1 = 0;
						int bezier_curve_idx_2 = 0;
						

						// Search for the first intersect
						for (int i_bezier = 0; i_bezier < 100; ++i_bezier) 
						{
							double t  = i_bezier / 99.0;
							double u  = 1 - t;
							double w1 = u*u*u;
							double w2 = 3*u*u*t;
							double w3 = 3*u*t*t;
							double w4 = t*t*t;
							double val_x_1 = w1*P[0].x + w2*P[1].x + w3*P[2].x + w4*P[3].x;
							double val_x_2 = w1*P[0].x + w2*(2.0*P[0].x-P[1].x) + w3*(2.0*P[3].x-P[2].x) + w4*P[3].x;
							double distance_1 = std::abs(val_x_1 - log1[i]);
							double distance_2 = std::abs(val_x_2 - log1[i]);

							if (distance_1 < tolerance_1)
							{
								tolerance_1 = distance_1;
								i_bezier_1 = i_bezier;
								bezier_curve_idx_1 = 1;
							}
							if (distance_2 < tolerance_1)
							{
								tolerance_1 = distance_2;
								i_bezier_1 = i_bezier;
								bezier_curve_idx_1 = 2;
							}
						}

						// Search for the second intersect
						for (int i_bezier = 0; i_bezier < 100; ++i_bezier) 
						{
							double t  = i_bezier / 99.0;
							double u  = 1 - t;
							double w1 = u*u*u;
							double w2 = 3*u*u*t;
							double w3 = 3*u*t*t;
							double w4 = t*t*t;
							double val_x_1 = w1*P[0].x + w2*P[1].x + w3*P[2].x + w4*P[3].x;
							double val_x_2 = w1*P[0].x + w2*(2.0*P[0].x-P[1].x) + w3*(2.0*P[3].x-P[2].x) + w4*P[3].x;
							double distance_1 = std::abs(val_x_1 - log1[i]);
							double distance_2 = std::abs(val_x_2 - log1[i]);
							
							if ((distance_1 < tolerance_2) & (i_bezier != i_bezier_1))
							{
								tolerance_2 = distance_1;
								i_bezier_2 = i_bezier;
								bezier_curve_idx_2 = 1;
							}
							if ((distance_2 < tolerance_2) & (i_bezier != i_bezier_1))
							{
								tolerance_2 = distance_2;
								i_bezier_2 = i_bezier;
								bezier_curve_idx_2 = 2;
							}
						}


						// Compute the first intersect y-coordinate
						double t  = i_bezier_1 / 99.0;
						double u  = 1 - t;
						double w1 = u*u*u;
						double w2 = 3*u*u*t;
						double w3 = 3*u*t*t;
						double w4 = t*t*t;
						double val_y_1;
						
						if(bezier_curve_idx_1==1)
							val_y_1 = w1*P[0].y + w2*P[1].y + w3*P[2].y + w4*P[3].y;
						else
							val_y_1 = w1*P[0].y + w2*(2.0*P[0].y-P[1].y) + w3*(2.0*P[3].y-P[2].y) + w4*P[3].y;

						// Compute the second intersect y-coordinate
						t  = i_bezier_2 / 99.0;
						u  = 1 - t;
						w1 = u*u*u;
						w2 = 3*u*u*t;
						w3 = 3*u*t*t;
						w4 = t*t*t;
						double val_y_2;

						if(bezier_curve_idx_2==1)
							val_y_2 = w1*P[0].y + w2*P[1].y + w3*P[2].y + w4*P[3].y;
						else
							val_y_2 = w1*P[0].y + w2*(2.0*P[0].y-P[1].y) + w3*(2.0*P[3].y-P[2].y) + w4*P[3].y;

						if((val_y_1-log2[i])*(val_y_2-log2[i])<=0)
						{
							log1Inner.push_back(log1[i]);
							log2Inner.push_back(log2[i]);
						}
					}
					
					ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(1,0,0,1), IMPLOT_AUTO, ImVec4(1,0,0,1));
					ImPlot::PlotScatter("##InnerPoints", &log1Inner[0], &log2Inner[0], log1Inner.size());

*/
/**
					// Drag Points on the perimeter of the Ellipse
					static double perimeterPoint_xMin = (log1Min+log1Max)/2.0 - 0.5*(log1Max-log1Min)/2.0;// Starting radius is half of the maximum radius			
					static double perimeterPoint_xMax = (log1Min+log1Max)/2.0 + 0.5*(log1Max-log1Min)/2.0;
					static double perimeterPoint_yMin = (log2Min+log2Max)/2.0 - 0.5*(log2Max-log2Min)/2.0;
					static double perimeterPoint_yMax = (log2Min+log2Max)/2.0 + 0.5*(log2Max-log2Min)/2.0;

					plotBezierCurve(perimeterPoint_xMin, perimeterPoint_xMax, perimeterPoint_yMin, perimeterPoint_yMax, log1Min, log1Max, log2Min, log2Max);
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
*/
				}


				ImPlot::EndPlot();
			}

			ImGui::EndChild();
		}

		ImGui::End();
	}
}


