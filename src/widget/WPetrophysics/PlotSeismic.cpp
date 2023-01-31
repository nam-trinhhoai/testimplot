/*
 *
 *
 *  Created on: 06 Sept 2022
 *      Author: l0359127
 */

#include "PlotSeismic.h"

PlotSeismic::PlotSeismic(WorkingSetManager* manager):
m_manager(manager)
{}

PlotSeismic::~PlotSeismic()
{}


// Seismic extraction.
std::pair<bool, PlotSeismic::IJKPoint> PlotSeismic::isPointInBoundingBox(Seismic3DAbstractDataset* dataset, WellUnit wellUnit, double logKey, WellBore* wellBore) 
{
	IJKPoint pt = {0,0,0};
	bool out = false;

	// get sampleI
	double sampleI;

	SampleUnit seismicUnit = dataset->cubeSeismicAddon().getSampleUnit();
	const AffineTransformation* sampleTransformSurrechantillon = dataset->sampleTransformation();	
	const Affine2DTransformation* ijToXYTransfo = dataset->ijToXYTransfo();
	int numTraces = dataset->width();
	int numProfils = dataset->depth();

	sampleI = wellBore->getDepthFromWellUnit(logKey, wellUnit, seismicUnit, &out);

	int numSamplesSurrechantillon = dataset->height();

	// check i
	if (out) {
		double i;
		sampleTransformSurrechantillon->indirect(sampleI, i);
		pt.i = i;

		out = (pt.i>=0) && (pt.i<numSamplesSurrechantillon);
	}

	// get and check jk
	if (out) {
		double x = wellBore->getXFromWellUnit(logKey, wellUnit, &out);
		double y;
		if (out) {
			y = wellBore->getYFromWellUnit(logKey, wellUnit, &out);
		}
		if (out) {
			double iMap, jMap;
			ijToXYTransfo->worldToImage(x, y, iMap, jMap);
			pt.j = iMap;
			pt.k = jMap;
			out = pt.j>=0 && pt.j<numTraces && pt.k>=0 && pt.k<numProfils;
		}
	}

	return std::pair<bool, IJKPoint>(out, pt);
}


// Plot the well logs from database in quickplot format
void PlotSeismic::showPlot() {
	// We demonstrate using the full viewport area or the work area (without menu-bars, task-bars etc.)
	// Based on your use case you may want one of the other.
	static bool use_work_area = true;
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);
	ImGui::SetNextWindowSize(use_work_area ? viewport->WorkSize : viewport->Size);

	// Get data from database
	WorkingSetManager::FolderList folders = m_manager->folders();

	// Etablish a list of wellbores that have logs from selected case study
	FolderData* wells = folders.wells;
	QList<IData*> iData = wells->data();

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

	// Only process if at least one wellbore was selected.
	int totalNumberOfWellBores = listWellBores.size();
	if(totalNumberOfWellBores > 0)
	{
		bool showPlot = true;
		ImGui::Begin("Plot a single log", &showPlot, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration);
		ImGui::BeginChild("ChildLeft", ImVec2(ImGui::GetContentRegionAvail().x*0.2f, -1), false, ImGuiWindowFlags_HorizontalScrollbar);

		// Get data
		std::vector<double> seismic_extract;
		std::vector<double> depth_seismic;
		std::string seismic_name = " ";


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
				WellBore* bore = listWellBores[selectedWell];

				// List of seismics
				{
					if (ImGui::TreeNode("Seismics"))
					{
						const Deviations& deviation = bore->deviations();
						double top = deviation.mds[0];
						double bottom = deviation.mds.back();

						//bool logIsSelected = bore->selectLog(0);
						//Logs currentLog = bore->currentLog();
						//double* depth = &currentLog.keys[0];
						//std::vector<double> keys = currentLog.keys;
						//int numPoints = currentLog.keys.size();

						static const int numPoints_seismic = 100000;
						double dDepth_seismic = (bottom-top)/numPoints_seismic; // TODO use 0.2 m, and remove duplicated points //;
						
						depth_seismic.clear();	
						for (int i=0; i<numPoints_seismic; i++)
						{
							depth_seismic.push_back(top + i*dDepth_seismic);
						}

						// Seismic data
						
						FolderData* seismics = folders.seismics;
						QList<IData*> iData_Seismic = seismics->data();

						int iData_idx = 0; // It seems like only one seismic survey can be selected at a time
					 	SeismicSurvey* seismicSurvey = dynamic_cast<SeismicSurvey*>(iData_Seismic[iData_idx]);
						QList<Seismic3DAbstractDataset*> dataset = seismicSurvey->datasets();

						const int numDatasets = dataset.size();

						if(numDatasets >0)
						{
							static int selected = -1;
							for (int n = 0; n < numDatasets; n++)
							{
								if (ImGui::Selectable(dataset[n]->name().toStdString().c_str(), selected == n))
									selected = n;
							}
							ImGui::TreePop();
						
							// Extract data from the selected seismic survey along the selected wellbore
							if(selected >-1)
							{
								bool out;
								IJKPoint pt, pt_previous;

								WellUnit wellUnit = MD;
								seismic_name = dataset[selected]->name().toStdString();
								seismic_extract.clear();

								for (int i=0; i<numPoints_seismic; i++)// Loop on all the points along wellbore
								{
									double logKey = depth_seismic[i];

									std::tie(out, pt) = isPointInBoundingBox(dataset[selected], wellUnit, logKey, bore);

									if(out) // only works for dataset dimV = 1
									{
										Seismic3DDataset * seismic3DDataset = dynamic_cast<Seismic3DDataset*>(dataset[selected]);

										float seismic_val; 

										seismic3DDataset->readSubTrace(&seismic_val, pt.i, pt.i+1, pt.j, pt.k, false);

										seismic_extract.push_back(seismic_val);
									}
									else
										seismic_extract.push_back(-999.25);
								}
							}
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
	
			//ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
			ImGui::BeginChild("ChildRight", ImVec2(-1, -1), true, window_flags);
			if (ImPlot::BeginPlot(seismic_name.c_str(), ImVec2(-1,-1))) 
			{
				//ImPlot::SetupAxesLimits(logMin, logMax, depthMin, depthMax);
				ImPlot::SetupAxis(ImAxis_X1,seismic_name.c_str());
				ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);	
				ImPlot::PlotLine(seismic_name.c_str(), &seismic_extract[0], &depth_seismic[0], seismic_extract.size());
				ImPlot::EndPlot();
			}
			ImGui::EndChild();
		}
		ImGui::End();
	}
}
