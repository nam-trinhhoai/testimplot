/*
 *
 *
 *  Created on: 12 Sept 2022
 *      Author: l0359127
 */

#include "PlotMultipleSeismicExtractionsAndWellLogs.h"

PlotMultipleSeismicExtractionsAndWellLogs::PlotMultipleSeismicExtractionsAndWellLogs(WorkingSetManager* manager):
ImGuiCore(),
m_manager(manager)
{}

PlotMultipleSeismicExtractionsAndWellLogs::~PlotMultipleSeismicExtractionsAndWellLogs()
{}

// Main code
void PlotMultipleSeismicExtractionsAndWellLogs::showPlot()
{	
	showPlot();	
}

// Seismic extraction.
std::pair<bool, PlotMultipleSeismicExtractionsAndWellLogs::IJKPoint> PlotMultipleSeismicExtractionsAndWellLogs::isPointInBoundingBox(Seismic3DAbstractDataset* dataset, WellUnit wellUnit, double logKey, WellBore* wellBore) 
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

// Interactive helper		
int PlotMultipleSeismicExtractionsAndWellLogs::interactiveHelper(Logs log)
{
	ImPlotPoint plotMousePos = ImPlot::GetPlotMousePos(IMPLOT_AUTO,IMPLOT_AUTO);

	// Identify the index of the current point
	double *depth, *logVal;
	depth = &log.keys[0];

	int idx = -1;
	
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

	return idx;
}

// Drag and drop to plot a log
template <typename T>
inline T RandomRange(T min, T max) {
    T scale = rand() / (T) RAND_MAX;
    return min + scale * ( max - min );
}

inline ImVec4 RandomColor() {
    ImVec4 col;
    col.x = RandomRange(0.0f,1.0f);
    col.y = RandomRange(0.0f,1.0f);
    col.z = RandomRange(0.0f,1.0f);
    col.w = 1.0f;
    return col;
}

// convenience struct to manage DND items; do this however you like
struct MyDndItem {
    int              Idx;
    int              Plt;
	int 			 chartIdx;
    ImVector<ImVec2> Data;
    ImVec4           Color;
    MyDndItem()        {
        static int i = 0;
        Idx = i++;
        Plt = 0;
		chartIdx = 0;

		Color = RandomColor();

		// tmp solution to activate ImGuiPayLoad
        Data.reserve(2);
        for (int k = 0; k < 2; ++k) {
            float t = k;
            Data.push_back(ImVec2(t, Idx));
        }
    }
    void Reset() { Plt = 0; chartIdx = 0; }
};

// Show the long crosshair cursor			
void PlotMultipleSeismicExtractionsAndWellLogs::longCrossHairCursor()
{
	int x, y;
	Uint32 buttons;

	SDL_PumpEvents();  // make sure we have the latest mouse state.

	buttons = SDL_GetMouseState(&x, &y);
	
	SDL_DisplayMode DM;
	SDL_GetCurrentDisplayMode(0, &DM);

	ImGuiWindow* window = ImGui::GetCurrentWindow();

	const ImGuiViewport* viewport = ImGui::GetMainViewport();

	window->DrawList->AddLine(ImVec2(0, y+viewport->WorkPos.y),ImVec2(DM.w, y+viewport->WorkPos.y), ImGui::GetColorU32(ImVec4(1, 0, 0, SDL_ALPHA_OPAQUE)), 1.0f);
	window->DrawList->AddLine(ImVec2(x+viewport->WorkPos.x, 0),ImVec2(x+viewport->WorkPos.x,DM.h), ImGui::GetColorU32(ImVec4(1, 0, 0, SDL_ALPHA_OPAQUE)), 1.0f);
}

// Drag and drop to plot a log
void PlotMultipleSeismicExtractionsAndWellLogs::showPlot() {
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

	FolderData* seismics = folders.seismics;
	QList<IData*> iData_Seismic = seismics->data();

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

	// Etablish a list of seismic dataset from selected case study
	std::vector<Seismic3DAbstractDataset*> listSeismicDatasets;

	int iDataSize_Seismic = iData_Seismic.size();

	if (iDataSize_Seismic > 0)
	{
		for (int i=0; i<iDataSize_Seismic; i++)
		{
			SeismicSurvey* seismicSurvey = dynamic_cast<SeismicSurvey*>(iData_Seismic[i]);
			QList<Seismic3DAbstractDataset*> dataset = seismicSurvey->datasets();
			const int numOfSeismicDatasets = dataset.size();

			if (numOfSeismicDatasets > 0)
			{
				for (int iSeismicDataset=0; iSeismicDataset < numOfSeismicDatasets; iSeismicDataset++)
				{
					listSeismicDatasets.push_back(dataset[iSeismicDataset]);	
				}
			}
		}
	}
	
	int totalNumberOfWellBores = listWellBores.size();
	if(totalNumberOfWellBores > 0)
	{		
		// Number of chart areas
		static int numChartAreas = 3;

		//static MyDndItem dnd[100]; // 100 is a limit number of logs
		static MyDndItem dnd_seismic[100];// 100 is a limit number of seismic extractions

		// Window
		static ImGuiWindowFlags winflags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;
		bool showPlot = true;
		ImGui::Begin("Plot a single log", &showPlot, winflags);

		// child window to serve as initial source for our DND items
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
		ImGui::BeginChild("DND_LEFT", ImVec2(ImGui::GetContentRegionAvail().x*0.2f, -1), false, window_flags);

		// Set mouse cursor
		if(ImGui::IsWindowHovered())		
			ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

		// Define arrays storing depth, log and seismic
		double *depth;
		double *log;
		int numLogs = 0; // intialization of numLogs here is mandatory
		int numSeismics = listSeismicDatasets.size();

		std::vector<std::string> logNames;
		std::vector<std::string> seismicNames;
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
		numPoints;
		
		// Checkbox to set linked depth axis
		static bool linkDepthAxis = true;
        ImGui::Checkbox("Link Depth", &linkDepthAxis);

		// Checkbox to set long crosshair
		static bool useLongCrossHair = false;
        ImGui::Checkbox("Long CrossHair", &useLongCrossHair);

		// Slider to set the number of chart areas
		ImGui::SliderInt("Plots", &numChartAreas, 1, 6);

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
				numLogs = bore->logsNames().size();

				// Get log names from selected well
				logNames.clear();

				for (int i=0; i<bore->logsNames().size();i++)
				{
					logNames.push_back( bore->logsNames()[i].toStdString() );	
				}
/**
				// List of logs
				{					
					if (ImGui::Button("Reset Logs")) {
						for (int k = 0; k < numLogs; ++k)
							dnd[k].Reset();
					}

					for (int k = 0; k < numLogs; ++k) {
						if (dnd[k].Plt > 0)
							continue;

						ImPlot::ItemIcon(dnd[k].Color); ImGui::SameLine();

						ImGui::Selectable(logNames[k].c_str(), false, 0, ImVec2(100, 0));

						if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
							ImGui::SetDragDropPayload("MY_DND", &k, sizeof(int));
							ImPlot::ItemIcon(dnd[k].Color); ImGui::SameLine();
							ImGui::TextUnformatted(logNames[k].c_str());
							ImGui::EndDragDropSource();
						}
					}
				}	
*/			
			}
		} // ImGui::TreeNode("Wellbores")

		// List of seismics
		if (ImGui::TreeNode("Seismics"))
		{					
			ImGui::TreePop();
			if (ImGui::Button("Reset Seismic")) {
				for (int k = 0; k < numSeismics; ++k)
					dnd_seismic[k].Reset();
			}

			for (int k = 0; k < numSeismics; ++k) {
				if (dnd_seismic[k].Plt > 0)
					continue;

				ImPlot::ItemIcon(dnd_seismic[k].Color); ImGui::SameLine();

				ImGui::Selectable(listSeismicDatasets[k]->name().toStdString().c_str(), false, 0, ImVec2(100, 0));

				if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
					ImGui::SetDragDropPayload("DND_SEISMIC", &k, sizeof(int)); //TODO this does not work!!!
					ImPlot::ItemIcon(dnd_seismic[k].Color); ImGui::SameLine();
					ImGui::TextUnformatted(seismicNames[k].c_str());
					ImGui::EndDragDropSource();
				}
			}
		}//ImGui::TreeNode("Seismics")	

		ImGui::EndChild();

		// Drag and Drop target
		if (ImGui::BeginDragDropTarget()) {
		    //if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MY_DND")) {
		    //    int i = *(int*)payload->Data; dnd[i].Reset();
		    //}

			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_SEISMIC")) {
		        int i = *(int*)payload->Data; dnd_seismic[i].Reset();
		    }
		    ImGui::EndDragDropTarget();
		}

		ImGui::SameLine();

		// Plot
		{
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
	
			ImGui::BeginChild("DND_RIGHT", ImVec2(-1, -1), true, window_flags);
			
			double chartWidth = ImGui::GetContentRegionAvail().x/numChartAreas;
			static ImPlotRect lims(0,1,10000,0);
	
			int point_idx;
			std::vector<int> logIdx;	
			for (int pltIdx=0; pltIdx<numChartAreas; pltIdx++)
			{
				// Plot
				if (ImPlot::BeginPlot(("##DND"+std::to_string(pltIdx)).c_str(), ImVec2(chartWidth,-1))) {
					ImPlot::SetupAxis(ImAxis_X1,NULL, ImPlotAxisFlags_Opposite);

					if(pltIdx==0)		
						ImPlot::SetupAxis(ImAxis_Y1,"Depth", ImPlotAxisFlags_Invert);
					else
						ImPlot::SetupAxis(ImAxis_Y1,NULL, ImPlotAxisFlags_Invert);

					//ImPlot::SetupAxesLimits(0, 100, depthMin, depthMax);

					if (linkDepthAxis)
					{
						ImPlot::SetupAxisLinks(ImAxis_Y1, &lims.Y.Max, &lims.Y.Min);
					
						if(pltIdx>0)
							ImPlot::SetupAxis(ImAxis_Y1,NULL, ImPlotAxisFlags_Invert | ImPlotAxisFlags_NoDecorations);
					}
/**
					if(numLogs >0)
					{
						for (int k = 0; k < numLogs; ++k) {
							if ((dnd[k].Plt == 1) & (dnd[k].chartIdx == pltIdx)) {
								// Update the list of plotted logs
								logIdx.push_back(k);

								ImPlot::SetNextLineStyle(dnd[k].Color);
				
								logName = logNames[k];

								logIsSelected = bore->selectLog(k);

								currentLog = bore->currentLog();
								numPoints = currentLog.attributes.size();

								log = &currentLog.attributes[0];
								depth = &currentLog.keys[0];

								ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);

								point_idx = interactiveHelper(currentLog);
							}
						}
					}
*/
					// allow the main plot area to be a DND target
					if (ImPlot::BeginDragDropTargetPlot()) {
						//if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MY_DND")) {
						//	int i = *(int*)payload->Data; dnd[i].Plt = 1; dnd[i].chartIdx = pltIdx;
						//}

						if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_SEISMIC")) {
							int i = *(int*)payload->Data; dnd_seismic[i].Plt = 1; dnd_seismic[i].chartIdx = pltIdx;
						}
						ImPlot::EndDragDropTarget();
					}
		
					// Show the long crosshair cursor	
					if(useLongCrossHair)
						longCrossHairCursor();

					ImPlot::EndPlot();
				}
				ImGui::SameLine();
			}

			// Tooltip showing the values at mouse position 
			if (linkDepthAxis & (logIdx.size()>0))
			{
				std::string toolTipString = "Current data: ";
				for (int i=0; i<logIdx.size(); i++)
				{
					int k = logIdx[i];

					logName = logNames[k];
					logIsSelected = bore->selectLog(k);

					currentLog = bore->currentLog();
					numPoints = currentLog.attributes.size();

					log = &currentLog.attributes[0];
					double val = log[point_idx];

					toolTipString += logName + "="+std::to_string(val)+", ";
				}
				ImGui::SetTooltip(toolTipString.c_str());
			}

			// Show the long crosshair cursor
			if(useLongCrossHair)	
				longCrossHairCursor();

			ImGui::EndChild();
		}

		ImGui::End();
	}
}


