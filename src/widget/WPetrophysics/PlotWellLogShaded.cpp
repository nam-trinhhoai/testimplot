/*
 *
 *
 *  Created on: 25 Jul 2022
 *      Author: l0359127
 */

#include "PlotWellLogShaded.h"

PlotWellLogShaded::PlotWellLogShaded(WorkingSetManager* manager):
m_manager(manager)
{}

PlotWellLogShaded::~PlotWellLogShaded()
{}

// Plot the well logs from database in quickplot format
void PlotWellLogShaded::showPlot() {
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

			if (ImPlot::BeginPlot(logName.c_str(), ImVec2(-1,-1))) 
			{
				//ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, alpha);
				//ImPlot::SetupAxesLimits(logMin, logMax, depthMin, depthMax);
				//ImPlot::SetupAxesLimits(depthMin, depthMax, logMin, logMax);
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

					ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);
					ImPlot::PlotLine("##Threshold", logThreshold, depth, numPoints);
					ImPlot::PlotShadedV("##Shale", logPositive, logThreshold, depth, numPoints, shaleColor);
					ImPlot::PlotShadedV("##Sand", logNegative, logThreshold, depth, numPoints, sandColor);

				}
				//ImPlot::PopStyleVar();
				ImPlot::EndPlot();
			}

			ImGui::EndChild();
		}

		ImGui::End();
	}
}

