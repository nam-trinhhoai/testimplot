/*
 *
 *
 *  Created on: 21 Sept 2022
 *      Author: l0359127
 */

#include "PlotWithMultipleKeys.h"
char* DndAxisLabel(WellUnit unit) {
	switch (unit)
	{
	case TVD:
		return "Verticle Depth";
		break;
	case TWT:
		return "Time";
		break;
	case MD:
		return "Depth";
		break;
	case UNDEFINED_UNIT:
		return "undefined unit";
		break;
	default:
		return "undefined";
		break;
	}
}
const char* getWellUnit(WellUnit wellUnit) {
	switch (wellUnit)
	{
	case TVD:
		return "TVD";
		break;
	case TWT:
		return "TWT";
		break;
	case MD:
		return "MD";
		break;
	case UNDEFINED_UNIT:
		return "UNDEFINED_UNIT";
		break;
	default:
		return "UNDEFINED";
		break;
	}
}
void colorPicker(ImGuiStyle& style) {
	static ImVec4 color = style.Colors[ImGuiCol_PlotLines];

	static bool saved_palette_init = true;
	static ImVec4 saved_palette[32] = { };
	if (saved_palette_init) {
		for (int n = 0; n < IM_ARRAYSIZE(saved_palette); n++) {
			ImGui::ColorConvertHSVtoRGB(n / 31.0f, 0.8f, 0.8f,
				saved_palette[n].x, saved_palette[n].y,
				saved_palette[n].z);
			saved_palette[n].w = 1.0f; // Alpha
		}
		saved_palette_init = false;
	}
	static ImVec4 backup_color;
	bool open_popup = ImGui::ColorButton("MyColor##3b", color);
	ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
	open_popup |= ImGui::Button("Pick Your PlotLine Color");
	if (open_popup) {
		ImGui::OpenPopup("mypicker");
		backup_color = color;
	}
	if (ImGui::BeginPopup("mypicker")) {
		ImGui::Text("MY CUSTOM COLOR PICKER WITH AN AMAZING PALETTE!");
		ImGui::Separator();
		ImGui::ColorPicker4("##picker", (float*)&color,
			ImGuiColorEditFlags_NoSidePreview
			| ImGuiColorEditFlags_NoSmallPreview);
		ImGui::SameLine();

		ImGui::BeginGroup(); // Lock X position
		ImGui::Text("Current");
		ImGui::ColorButton("##current", color,
			ImGuiColorEditFlags_NoPicker
			| ImGuiColorEditFlags_AlphaPreviewHalf,
			ImVec2(60, 40));
		ImGui::Text("Previous");
		if (ImGui::ColorButton("##previous", backup_color,
			ImGuiColorEditFlags_NoPicker
			| ImGuiColorEditFlags_AlphaPreviewHalf,
			ImVec2(60, 40)))
			color = backup_color;
		ImGui::Separator();
		ImGui::Text("Pick Your Color");
		for (int n = 0; n < IM_ARRAYSIZE(saved_palette); n++) {
			ImGui::PushID(n);
			if ((n % 8) != 0)
				ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.y);

			ImGuiColorEditFlags palette_button_flags =
				ImGuiColorEditFlags_NoAlpha
				| ImGuiColorEditFlags_NoPicker
				| ImGuiColorEditFlags_NoTooltip;
			if (ImGui::ColorButton("##palette", saved_palette[n],
				palette_button_flags, ImVec2(20, 20)))
				color = ImVec4(saved_palette[n].x, saved_palette[n].y,
					saved_palette[n].z, color.w); // Preserve alpha!

			// Allow user to drop colors into each palette entry. Note that ColorButton() is already a
			// drag source by default, unless specifying the ImGuiColorEditFlags_NoDragDrop flag.
			if (ImGui::BeginDragDropTarget()) {
				if (const ImGuiPayload* payload =
					ImGui::AcceptDragDropPayload(
						IMGUI_PAYLOAD_TYPE_COLOR_3F))
					memcpy((float*)&saved_palette[n], payload->Data,
						sizeof(float) * 3);
				if (const ImGuiPayload* payload =
					ImGui::AcceptDragDropPayload(
						IMGUI_PAYLOAD_TYPE_COLOR_4F))
					memcpy((float*)&saved_palette[n], payload->Data,
						sizeof(float) * 4);
				ImGui::EndDragDropTarget();
			}

			ImGui::PopID();
		}
		style.Colors[ImGuiCol_PlotLines] = color;
		ImGui::EndGroup();
		ImGui::EndPopup();
	}
}

PlotWithMultipleKeys::PlotWithMultipleKeys(WorkingSetManager* manager) :
	m_manager(manager)
{
	// Get data from database
	WorkingSetManager::FolderList folders = m_manager->folders();
	wells = folders.wells;
	iData = wells->data();

	seismics = folders.seismics;
	iData_Seismic = seismics->data();

	selectedWellUnit = WellUnit::MD;
	numChartAreas = 3;
	int iDataSize = iData.size();
	total_logs_count = 0;
	if (iDataSize > 0)
	{
		for (int i = 0; i < iDataSize; i++)
		{
			WellHead* wellHead = dynamic_cast<WellHead*>(iData[i]);

			int numberOfWellBores = wellHead->wellBores().size();

			if (!wellHead->wellBores().empty())
			{
				for (int iWellbore = 0; iWellbore < numberOfWellBores; iWellbore++)
				{
					WellBore* bore = wellHead->wellBores()[iWellbore];

					bool hasLogs = (bore->logsNames().size() > 0);
					total_logs_count += bore->logsNames().size();
					// Only add wellbores that have logs to the list
					if (hasLogs)
						listWellBores.push_back(bore);
				}
			}
		}
	}

	int iDataSize_Seismic = iData_Seismic.size();

	if (iDataSize_Seismic > 0)
	{
		for (int i = 0; i < iDataSize_Seismic; i++)
		{
			SeismicSurvey* seismicSurvey = dynamic_cast<SeismicSurvey*>(iData_Seismic[i]);
			QList<Seismic3DAbstractDataset*> dataset = seismicSurvey->datasets();
			const int numOfSeismicDatasets = dataset.size();

			if (numOfSeismicDatasets > 0)
			{
				for (int iSeismicDataset = 0; iSeismicDataset < numOfSeismicDatasets; iSeismicDataset++)
				{
					listSeismicDatasets.push_back(dataset[iSeismicDataset]);
				}
			}
		}
	}

	logNameOnChart = new LogNameOnChart(total_logs_count);
	processed_logs_ptr = new processed_log[total_logs_count];


	// An alternative for the loop above,
	// But this will make all the log have the same color
	// processed_logs = std::vector<processed_log>(total_logs_count, processed_log());
		int cur_log_idx = 0;
	for (int i = 0; i < listWellBores.size(); i++) {
		WellBore* current_bore = listWellBores[i];
		int num_logs = current_bore->logsNames().size();
		for (int j = 0; j < num_logs; j++) {
			current_bore->selectLog(j);
			Logs current_log = current_bore->currentLog();
			processed_logs_ptr[cur_log_idx].update(*current_bore, current_log, j);
			cur_log_idx++;

		}
	}
}

PlotWithMultipleKeys::~PlotWithMultipleKeys()
{
	delete[] processed_logs_ptr;
}
void PlotWithMultipleKeys::update_processed_logs_chart_idx(int idx, std::string lName) {

	for (int i = 0; i < total_logs_count; i++) {
		std::string name = processed_logs_ptr[i].log_name.toStdString();
		if (lName.compare(name) == 0) {
			processed_logs_ptr[i].update_chart_idx(idx);
		}
	}
}

void PlotWithMultipleKeys::setting(ImGuiStyle& style) {

	const char* combo_preview_value = getWellUnit(selectedWellUnit);  // Pass in the preview value visible before opening the combo (it could be anything)

	ImGui::Checkbox("Link Depth", &linkDepthAxis);

	// Checkbox to set long crosshair

	ImGui::Checkbox("Long CrossHair", &useLongCrossHair);


	// Slider to set the number of chart areas
	ImGui::SliderInt("Plots", &numChartAreas, 1, 6);

	ImGui::Checkbox("Anti-aliased lines", &style.AntiAliasedLines);

	// Dropdown to choose well unit
	if (ImGui::BeginCombo("Well unit", combo_preview_value, flags))
	{
		for (int n = 0; n < 3; n++)
		{
			const bool is_selected = (selectedWellUnit == WellUnit(n));
			if (ImGui::Selectable(getWellUnit(WellUnit(n)), is_selected))
				selectedWellUnit = WellUnit(n);

			// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	ImGui::ShowFontSelector("Font Selector");
	colorPicker(style);
}
// Seismic extraction.
std::pair<bool, PlotWithMultipleKeys::IJKPoint> PlotWithMultipleKeys::isPointInBoundingBox(Seismic3DAbstractDataset* dataset, WellUnit wellUnit, double logKey, WellBore* wellBore)
{
	IJKPoint pt = { 0,0,0 };
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

	if (out) {
		double i;
		sampleTransformSurrechantillon->indirect(sampleI, i);
		pt.i = i;

		out = (pt.i >= 0) && (pt.i < numSamplesSurrechantillon);
	}

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
			out = pt.j >= 0 && pt.j < numTraces&& pt.k >= 0 && pt.k < numProfils;
		}
	}

	return std::pair<bool, IJKPoint>(out, pt);
}

// Interactive helper		
int PlotWithMultipleKeys::interactiveHelper(double* depth)
{
	ImPlotPoint plotMousePos = ImPlot::GetPlotMousePos(IMPLOT_AUTO, IMPLOT_AUTO);

	// Identify the index of the current point

	int idx = -1;

	double tolerance = 1e6;
	int numPoints = sizeof(depth);
	for (int i = 0; i < numPoints; i++)
	{
		double distance = (depth[i] - plotMousePos.y) * (depth[i] - plotMousePos.y);

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
	T scale = rand() / (T)RAND_MAX;
	return min + scale * (max - min);
}

inline ImVec4 RandomColor() {
	ImVec4 col;
	col.x = RandomRange(0.0f, 1.0f);
	col.y = RandomRange(0.0f, 1.0f);
	col.z = RandomRange(0.0f, 1.0f);
	col.w = 1.0f;
	return col;
}
//Mapping WellUnit to corresponding char*

// Show the long crosshair cursor			
void PlotWithMultipleKeys::longCrossHairCursor()
{
	ImGuiWindow* window = ImGui::GetCurrentWindow();

	ImVec2 mousePos = ImGui::GetMousePos();

	const ImGuiViewport* viewport = ImGui::GetMainViewport();

	window->DrawList->AddLine(ImVec2(0, mousePos.y + viewport->WorkPos.y), ImVec2(viewport->WorkSize.x, mousePos.y + viewport->WorkPos.y), ImGui::GetColorU32(ImVec4(1, 0, 0, 255)), 1.0f);
	window->DrawList->AddLine(ImVec2(mousePos.x + viewport->WorkPos.x, 0), ImVec2(mousePos.x + viewport->WorkPos.x, viewport->WorkSize.y), ImGui::GetColorU32(ImVec4(1, 0, 0, 255)), 1.0f);
}

// Drag and drop to plot a log
void PlotWithMultipleKeys::showPlot() {
	// We demonstrate using the full viewport area or the work area (without menu-bars, task-bars etc.)
	// Based on your use case you may want one of the other.
	static bool use_work_area = true;
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);
	ImGui::SetNextWindowSize(use_work_area ? viewport->WorkSize : viewport->Size);



	int totalNumberOfWellBores = listWellBores.size();
	if (totalNumberOfWellBores > 0)
	{

		// Window
		static ImGuiWindowFlags winflags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;
		bool showPlot = true;
		ImGui::Begin("Plot a single log", &showPlot, winflags);

		// child window to serve as initial source for our DND items
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
		ImGui::BeginChild("DND_LEFT", ImVec2(ImGui::GetContentRegionAvail().x * 0.2f, -1), false, window_flags);

		// Set mouse cursor
		if (ImGui::IsWindowHovered())
			ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

		// Define arrays storing depth, log and seismic
		double* log;
		int numLogs = 0; // intialization of numLogs here is mandatory
		int numSeismics = listSeismicDatasets.size();

		std::vector<std::string> logNames;
		std::vector<std::string> seismicNames;
		std::vector<double> seismic_extract;
		std::vector<double> depth_seismic;
		static const double dDepth_seismic = 0.2; //we consider a step of 0.2 meter per seismic points.
		static int numPoints_seismic;

		std::string logName = " ";
		float const nullValue = -999.25;
		float logMin = 1e6;
		float logMax = 1e-6;
		float depthMin = 1e6;
		float depthMax = 1e-6;

		// Initialization
		WellBore* bore;
		bool logIsSelected;
		Logs currentLog;
		static WellUnit keyUnit = MD;
		char* dndAxisLabel = DndAxisLabel(keyUnit);
		// Checkbox to set linked depth axis
		ImGuiStyle& style = ImGui::GetStyle();
		setting(style);

		static int selectedWell = -1;
		if (ImGui::TreeNode("WellBores"))
		{
			// Select a well
			for (int n = 0; n < totalNumberOfWellBores; n++)
			{
				// WellUnit TWT is choosen, checking for compatible
				if (selectedWellUnit == WellUnit::TWT && !listWellBores[n]->isWellCompatibleForTime(true)) {
					// Reset selected well
					if (selectedWell == n) {
						selectedWell = -1;
					}
					continue;
				}
				if (ImGui::Selectable(listWellBores[n]->name().toStdString().c_str(), selectedWell == n))
					selectedWell = n;
			}
			ImGui::TreePop();

			if (selectedWell > -1)
			{
				bore = listWellBores[selectedWell];

				numLogs = bore->logsNames().size();
				if (ImGui::Button("Reset Logs")) {
					for (int k = 0; k < total_logs_count; k++) {
						processed_logs_ptr[k].reset();

					}
					logNameOnChart->reset();
				}
				for (int i = 0; i < total_logs_count; i++) {
					processed_log* p_log = &processed_logs_ptr[i];
					if (p_log->chart_idx != -1) {
						continue;
					}
					else if (bore == p_log->wellbore) {
						ImPlot::ItemIcon(p_log->color); ImGui::SameLine();
						ImGui::Selectable(p_log->log_name.toStdString().c_str(), false, 0, ImVec2(100, 0));
						if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
							ImGui::SetDragDropPayload("MY_DND", &i, sizeof(int));
							ImPlot::ItemIcon(p_log->color); ImGui::SameLine();
							ImGui::TextUnformatted(p_log->log_name.toStdString().c_str());
							ImGui::EndDragDropSource();
						}
					}
				}

				// Populate points along the selected wellbore trajectory to extract seismic
				const Deviations& deviation = bore->deviations();
				double top = deviation.mds[0];
				double bottom = deviation.mds.back();

				numPoints_seismic = (int)(bottom - top) / dDepth_seismic;

				depth_seismic.clear();
				for (int i = 0; i < numPoints_seismic; i++)
				{
					depth_seismic.push_back(top + i * dDepth_seismic);
				}
			}
		} // ImGui::TreeNode("Wellbores")

		// List of seismics
		//if (ImGui::TreeNode("Seismics"))
		//{
		//	ImGui::TreePop();

		//	if (ImGui::Button("Reset Seismic")) {
		//		for (int k = 0; k < numSeismics; ++k)
		//			dnd_seismic[k].Reset();
		//	}

		//	for (int k = 0; k < numSeismics; ++k) {
		//		if (dnd_seismic[k].Plt > 0)
		//			continue;

		//		ImPlot::ItemIcon(dnd_seismic[k].Color); ImGui::SameLine();

		//		std::string datasetName = listSeismicDatasets[k]->name().toStdString();

		//		ImGui::Selectable(datasetName.c_str(), false, 0, ImVec2(100, 0));

		//		if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
		//			ImGui::SetDragDropPayload("DND_SEISMIC", &k, sizeof(int)); //TODO this does not work!!!
		//			ImPlot::ItemIcon(dnd_seismic[k].Color); ImGui::SameLine();
		//			ImGui::TextUnformatted(datasetName.c_str());
		//			ImGui::EndDragDropSource();
		//		}
		//	}
		//}//ImGui::TreeNode("Seismics")

		ImGui::EndChild();

		// Drag and Drop target
		if (ImGui::BeginDragDropTarget()) {
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MY_DND")) {
				int i = *(int*)payload->Data; processed_logs_ptr[i].reset();

			}

			//if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_SEISMIC")) {
			//	int i = *(int*)payload->Data; dnd_seismic[i].Reset();
			//}
			ImGui::EndDragDropTarget();
		}

		ImGui::SameLine();

		// Plot
		{
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;

			ImGui::BeginChild("DND_RIGHT", ImVec2(-1, -1), true, window_flags);

			double chartWidth = ImGui::GetContentRegionAvail().x / numChartAreas;
			static ImPlotRect lims(0, 1, 10000, 0);

			int point_idx, seismic_point_idx;
			std::vector<int> logIdx;
			std::vector<int> seismicIdx;
			for (int pltIdx = 0; pltIdx < numChartAreas; pltIdx++)
			{
				// Plot
				if (ImPlot::BeginPlot(("##DND" + std::to_string(pltIdx)).c_str(), ImVec2(chartWidth, -1))) {
					ImPlot::SetupAxis(ImAxis_X1, NULL, ImPlotAxisFlags_Opposite | ImPlotAxisFlags_AutoFit);

					if (pltIdx == 0)
						ImPlot::SetupAxis(ImAxis_Y1, getWellUnit(selectedWellUnit), ImPlotAxisFlags_Invert | ImPlotAxisFlags_AutoFit);
					else
						ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_Invert | ImPlotAxisFlags_AutoFit);

					if (linkDepthAxis)
					{
						ImPlot::SetupAxisLinks(ImAxis_Y1, &lims.Y.Max, &lims.Y.Min);
						if (pltIdx > 0)
							ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_Invert | ImPlotAxisFlags_NoDecorations);
					}

					if (total_logs_count > 0)
					{
						for (int k = 0; k < total_logs_count; ++k) {
							processed_log* p_log = &processed_logs_ptr[k];
							if (processed_logs_ptr[k].chart_idx == pltIdx && p_log->wellbore == bore) {

								if (p_log->cur_unit != selectedWellUnit) {
									p_log->update_keys_on_unit(selectedWellUnit);
								}
								ImPlot::SetNextLineStyle(p_log->color);
								ImPlot::PlotLine(p_log->log_name.toStdString().c_str(),
									p_log->attributes + p_log->start,
									p_log->keys + p_log->start, p_log->end - p_log->start);
								point_idx = interactiveHelper(p_log->keys);

							}
						}
					}

					// Plot seismic
					//if (selectedWell > -1)
					//{
					//	if (numSeismics > 0)
					//	{
					//		for (int k = 0; k < numSeismics; ++k) {
					//			if ((dnd_seismic[k].Plt == 1) & (dnd_seismic[k].chartIdx == pltIdx)) {
					//				// Update the list of plotted logs
					//				seismicIdx.push_back(k);

					//				bool out;
					//				IJKPoint pt, pt_previous;

					//				WellUnit wellUnit = MD;
					//				std::string seismic_name = listSeismicDatasets[k]->name().toStdString();
					//				seismic_extract.clear();
					//				std::vector<double> depth_seismic_update;// This vector stores only data of one point per cell.

					//				bool haseFirstPoint = false;// To verify if there is at least one point.

					//				for (int i = 0; i < numPoints_seismic; i++)// Loop on all the seismic extraction points along wellbore
					//				{
					//					double logKey = depth_seismic[i];

					//					std::tie(out, pt) = isPointInBoundingBox(listSeismicDatasets[k], wellUnit, logKey, bore);

					//					// We consider only one point in each cell along the wellbore trajectory.
					//					if (out && haseFirstPoint && (pt.i == pt_previous.i) && (pt.j == pt_previous.j) && (pt.k == pt_previous.k))
					//						continue;
					//					else if (out)
					//					{
					//						pt_previous = pt;
					//						haseFirstPoint = true;
					//					}

					//					if (out) // only works for dataset dimV = 1
					//					{
					//						Seismic3DDataset* seismic3DDataset = dynamic_cast<Seismic3DDataset*>(listSeismicDatasets[k]);

					//						float seismic_val;

					//						seismic3DDataset->readSubTrace(&seismic_val, pt.i, pt.i + 1, pt.j, pt.k, false);

					//						seismic_extract.push_back(seismic_val);
					//						depth_seismic_update.push_back(logKey);
					//					}
					//				}

					//				ImPlot::SetNextLineStyle(dnd_seismic[k].Color);
					//				ImPlot::PlotLine(seismic_name.c_str(), &seismic_extract[0], &depth_seismic_update[0], seismic_extract.size());

					//				seismic_point_idx = interactiveHelper(&depth_seismic_update[0]);
					//			}
					//		}
					//	}
					//}

					// allow the main plot area to be a DND target
					if (ImPlot::BeginDragDropTargetPlot()) {
						if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MY_DND")) {
							int i = *(int*)payload->Data;
							std::string lName = processed_logs_ptr[i].log_name.toStdString();
							logNameOnChart->add_logNamesOnChart(lName);
							update_processed_logs_chart_idx(pltIdx, lName);

						}

						//if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_SEISMIC")) {
						//	int i = *(int*)payload->Data; dnd_seismic[i].Plt = 1; dnd_seismic[i].chartIdx = pltIdx;
						//}
						ImPlot::EndDragDropTarget();
					}

					// Show the long crosshair cursor	
					if (useLongCrossHair)
						longCrossHairCursor();

					ImPlot::EndPlot();
				}
				ImGui::SameLine();
			}

			// Tooltip showing the values at mouse position 
			//if (linkDepthAxis & (logIdx.size() > 0))
			//{
			//	std::string toolTipString = "Current data: ";
			//	for (int i = 0; i < logIdx.size(); i++)
			//	{
			//		// Get log values at mouse position
			//		int k = logIdx[i];

			//		logName = logNames[k];
			//		logIsSelected = bore->selectLog(k);

			//		currentLog = bore->currentLog();
			//		numPoints = currentLog.attributes.size();

			//		log = &currentLog.attributes[0];
			//		double val = log[point_idx];

			//		toolTipString += logName + "=" + std::to_string(val) + ", ";

			//	}
			//	for (int i = 0; i < seismicIdx.size(); i++)
			//	{
			//		// Get seismic values at mouse position
			//		int k = seismicIdx[i];

			//		std::string seismicName = listSeismicDatasets[k]->name().toStdString();


			//		toolTipString += seismicName + ", ";

			//	}

			//	ImGui::SetTooltip(toolTipString.c_str());
			//}

			// Show the long crosshair cursor
			if (useLongCrossHair)
				longCrossHairCursor();

			ImGui::EndChild();
		}

		ImGui::End();
	}
}

