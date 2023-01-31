/*
 *
 *
 *  Created on: 26 Jul 2022
 *      Author: l0359127
 */

#include "PlotWellLogInteractive.h"

PlotWellLogInteractive::PlotWellLogInteractive(WorkingSetManager* manager):
m_manager(manager)
{}

PlotWellLogInteractive::~PlotWellLogInteractive()
{}

// Interactive helper		
std::tuple<int, bool> PlotWellLogInteractive::interactiveHelper(Logs log)
{
	int x, y;
	
	ImGuiContext& g = *GImGui;
	const ImGuiStyle& style = g.Style;

	ImGuiWindow* window = ImGui::GetCurrentWindow();

	ImVec2 wPos = ImGui::GetWindowPos();
	ImVec2 wSize = ImGui::GetWindowSize();
	ImVec2 fPadding = style.FramePadding;
	ImVec2 innerSpacing = style.ItemInnerSpacing;

	//window->DrawList->AddLine(ImVec2(wPos.x, y),ImVec2(wPos.x+wSize.x, y), ImGui::GetColorU32(ImVec4(255, 255, 255, SDL_ALPHA_OPAQUE)), 1.0f);
	//window->DrawList->AddLine(ImVec2(x, wPos.y),ImVec2(x, wPos.y+wSize.y), ImGui::GetColorU32(ImVec4(255, 255, 255, SDL_ALPHA_OPAQUE)), 1.0f);

	//window->DrawList->AddLine(ImVec2(0, y),ImVec2(DM.w, y), ImGui::GetColorU32(ImVec4(255, 255, 255, SDL_ALPHA_OPAQUE)), 1.0f);
	//window->DrawList->AddLine(ImVec2(x, 0),ImVec2(x,DM.h), ImGui::GetColorU32(ImVec4(255, 255, 255, SDL_ALPHA_OPAQUE)), 1.0f);

	ImPlotPoint plotMousePos = ImPlot::GetPlotMousePos(IMPLOT_AUTO,IMPLOT_AUTO);

	// Identify the index of the current point
	double *depth, *logVal;
	depth = &log.keys[0];

	int idx = -1;
	bool curveIsSelected = false;

	double tolerance = 1e6;
	int numPoints = log.attributes.size();
	for (int i=0; i<numPoints; i++)
	{
		double distance = (depth[i] - plotMousePos.y)*(depth[i] - plotMousePos.y);
		
		if (distance < tolerance)
		{
			idx = i;
			tolerance = distance;		
		}	
	}

	// Show the values at mouse position
	ImGui::SetTooltip("idx=%d, x=%2f, y=%2f", idx, plotMousePos.x, plotMousePos.y);

	// Verify if the mouse is close enable to the curve to hightlight it
	logVal = &log.attributes[0];	

	double logMax = 1e-6;
	double logMin = 1e6;
	double nullValue = -999.25;

	for (int i=0; i<numPoints; i++)
	{	
		if (logVal[i] > nullValue)
		{
			if (logVal[i] > logMax)
				logMax = logVal[i];
			if (logVal[i] < logMin)
				logMin = logVal[i];
		}
	}

	// Estimate the tolerance using min and max values
	double valTolerance = 0.01*(logMax-logMin);
	double valDistance = std::abs(logVal[idx] - plotMousePos.x);

	if (valDistance < valTolerance)
		curveIsSelected = true;

	return std::make_tuple(idx, curveIsSelected);
}


// Plot the well logs from database in quickplot format
void PlotWellLogInteractive::showPlot() {
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

	int totalNumberOfWellBores = listWellBores.size();
	if(totalNumberOfWellBores > 0)
	{
		static ImGuiWindowFlags winflags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;
		bool showPlot = true;

		ImGui::Begin("Plot a single log", &showPlot, winflags);
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
		ImGui::BeginChild("ChildLeft", ImVec2(ImGui::GetContentRegionAvail().x*0.2f, -1), false, window_flags);


		// Get data
		double *depth;
		double *log;
		int numLogs = 0;

		std::string logName = " ";
		float const nullValue = -999.25;
		float logMin = 1e6;
		float logMax = 1e-6;
		float depthMin = 1e6;
		float depthMax = 1e-6;

		// Initialization
		WellBore *bore;
		bool logIsSelected;
		Logs currentLog;
		numPoints = 0;
		
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
				numLogs = bore->logsNames().size();


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

		} // ImGui::TreeNode("Wells")

		ImGui::EndChild();

		ImGui::SameLine();

		// Plot
		{
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
	
			//ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
			ImGui::BeginChild("ChildRight", ImVec2(-1, -1), true, window_flags);

			static float alpha = 0.25f;
    		ImGui::DragFloat("Alpha",&alpha,0.01f,0,1);

			static float threshold = 50;
    		ImGui::DragFloat("Threshold",&threshold, 1, 0, 200);

			ImU32 sandColor = ImGui::ColorConvertFloat4ToU32(ImVec4(1, 1, 0, alpha));
			ImU32 shaleColor = ImGui::ColorConvertFloat4ToU32(ImVec4(0, 102.0f/255, 51.0f/255, alpha));
			
			const char* label = logName.c_str();
			ImVec2 frame_size = ImVec2(0,80.0f);

			if (ImPlot::BeginPlot(label, ImVec2(-1,-1))) 
			{
				ImPlot::SetupAxis(ImAxis_X1,logName.c_str());
				ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);	
				
				if(currentLog.attributes.size()>0)
				{		

					double logThreshold[numPoints];
					double logPositive[numPoints];
					double logNegative[numPoints];

					for (int i=0; i<numPoints; i++)
					{
						logThreshold[i] = threshold;

						if(log[i] > threshold)
						{
							logPositive[i] = log[i];
							logNegative[i] = threshold;
						}
						else
						{
							logPositive[i] = threshold;
							logNegative[i] = log[i];
						}
					}

					ImPlot::SetNextLineStyle(ImVec4(0,0,1,alpha), 0.5);
					ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);

					ImPlot::SetNextLineStyle(ImVec4(55,55,55,alpha), 0.5);
					ImPlot::PlotLine("##Threshold", logThreshold, depth, numPoints);

					ImPlot::PlotShadedV("##Shale", logPositive, logThreshold, depth, numPoints, shaleColor);
					ImPlot::PlotShadedV("##Sand", logNegative, logThreshold, depth, numPoints, sandColor);

					// Plot active point for interactive animation
					int idx = -1;
					bool curveIsSelected = false;

					std::tie(idx, curveIsSelected) = interactiveHelper(currentLog);

					double activeDepth[1] = {depth[idx]};
					double activeLog[1] = {log[idx]};

					ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 1.0f);
					ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(1,0,0,1), IMPLOT_AUTO, ImVec4(1,0,0,1));
					ImPlot::PlotScatter("##ActivePoint", activeLog, activeDepth, 1);
					ImPlot::PopStyleVar();

					if (curveIsSelected)
					{
						ImPlot::SetNextLineStyle(ImVec4(1,0,0,1), 1);
						ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);
					}
					else
					{
						ImPlot::SetNextLineStyle(ImVec4(0,0,1,alpha), 0.5);
						ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);
					}
				}

				ImPlot::EndPlot();
			}

			ImGui::EndChild();
		}

		ImGui::End();
	}
}

