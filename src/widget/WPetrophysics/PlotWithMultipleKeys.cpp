/*
 *
 *
 *  Created on: 21 Sept 2022
 *      Author: l0359127
 */

#include "PlotWithMultipleKeys.h"

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


PlotWithMultipleKeys::PlotWithMultipleKeys(WorkingSetManager* manager) :
	m_manager(manager)
{
	// Get data from database
	//temporarly work on charts
	charts.clear();
	charts.reserve(12);
	for (int i = 0; i < 3; i++) {
		chart* temp = new chart();
		charts.push_back(temp);
	}
	WorkingSetManager::FolderList folders = m_manager->folders();
	wells = folders.wells;
	iData = wells->data();

	seismics = folders.seismics;
	iData_Seismic = seismics->data();

	selectedWellUnit = WellUnit::MD;
	int iDataSize = iData.size();
	total_logs_count = 0;
	background_color = ImVec4(0.9, 0.9, 0.9, 1.0);
	if (iDataSize > 0)
	{
		for (int i = 0; i < iDataSize; i++)
		{
			WellHead* wellHead = dynamic_cast<WellHead*>(iData[i]);

			int numberOfWellBores = wellHead->wellBores().size();
			listWellBores.reserve(numberOfWellBores);
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
	processed_logs_ptr = new log_data[total_logs_count];
	// An alternative for the loop above,
	// But this will make all the log have the same color
	int cur_log_idx = 0;
	linkDepthAxis = new bool[listWellBores.size()];
	for (int i = 0; i < listWellBores.size(); i++) {
		WellBore* current_bore = listWellBores[i];
		linkDepthAxis[i] = true;
		int num_logs = current_bore->logsNames().size();
		for (int j = 0; j < num_logs; j++) {
			processed_logs_ptr[cur_log_idx].update(*current_bore, &m_logViews, j);
			cur_log_idx++;
		}
	}
	trackRule = new track_rule(3, MD);
	lims = new ImPlotRect(0, 1, 0, 2000);
}

PlotWithMultipleKeys::~PlotWithMultipleKeys()
{
	delete[] processed_logs_ptr;
	delete[] linkDepthAxis;
	m_logViews.clear();
	for (int i = 0; i < charts.size(); i++) {
		delete charts[i];
	}
}
log_view* PlotWithMultipleKeys::findLogViewByLogname(QString lname) {
	return m_logViews.find(lname) != m_logViews.end() ? m_logViews.find(lname)->second : nullptr;
}

void PlotWithMultipleKeys::addLogInChart(int idx, QString lname) {

	log_view* lview = findLogViewByLogname(lname);
	lview->update_chart_idx(charts[idx]);
	charts[idx]->add_log(lview);
}
void PlotWithMultipleKeys::removeLog(chart* chart, log_data* curLog) {

	chart->remove_log(curLog->logView);
	curLog->logView->reset();
}
void PlotWithMultipleKeys::menubar(ImGuiStyle& style) {
	const char* combo_preview_value = getWellUnit(selectedWellUnit);  // Pass in the preview value visible before opening the combo (it could be anything)

	ImGui::Checkbox("Link Depth", &linkDepthAxis[selectedWell]);

	// Checkbox to set long crosshair
	ImGui::SameLine();
	ImGui::Checkbox("Long CrossHair", &useLongCrossHair);

	ImGui::SameLine();
	ImGui::Button("Add Track test");
	if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
		ImGui::SetDragDropPayload("ADD_TRACK", NULL, 0);
		ImGui::Text("Use this to Add track");
		ImGui::EndDragDropSource();
	}
	ImGui::SameLine();
	if (ImGui::Button("Remove selected charts")) {
		std::vector<chart*>::iterator it;
		for (it = charts.begin(); it != charts.end(); ) {
			if ((*it)->selected) {
				delete* it;
				it = charts.erase(it);
			}
			else {
				++it;
			}
		}
		resetSelectedChart();
	}

	ImGui::SameLine();
	ImGui::SliderFloat("Plots Shade Opacity", &opacity, 0.0f, 1.0f);

	ImGui::SameLine();
	ImGui::Checkbox("Anti-aliased lines", &style.AntiAliasedLines);


	ImGui::SameLine();
	// Dropdown to choose well unit
	if (ImGui::BeginCombo("Well unit", combo_preview_value, flags))
	{
		for (int n = 0; n < 3; n++)
		{
			const bool is_selected = (selectedWellUnit == WellUnit(n));
			if (ImGui::Selectable(getWellUnit(WellUnit(n)), is_selected)) {
				selectedWellUnit = WellUnit(n);
				trackRule->changeWellUnit(selectedWellUnit);
			}
			// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}

	ImGui::SameLine();
	colorPicker(&background_color, "Background Color");

	ImGui::SameLine();
	ImGui::ShowFontSelector("Font Selector");


	ImGui::SameLine();
	ImGui::SliderFloat("Line Thickness", &line_weight, 0.0f, 3.0f, "ratio = %.2f");


	ImGui::SameLine();
	ImGui::DragFloat("Track width", &chart_width, 5.0f, 200.0f, 500.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);


	ImGui::SameLine();
	if (ImGui::DragFloat("Range limit", &r_limit, 10.0f, 200.0f, 2000.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
		r_change = true;
	}
}
void PlotWithMultipleKeys::setting(ImGuiStyle& style) {
	const char* combo_preview_value = getWellUnit(selectedWellUnit);  // Pass in the preview value visible before opening the combo (it could be anything)

	ImGui::Checkbox("Link Depth", &linkDepthAxis[selectedWell]);

	// Checkbox to set long crosshair
	ImGui::Checkbox("Long CrossHair", &useLongCrossHair);


	if (ImGui::Button("Remove selected charts")) {
		std::vector<chart*>::iterator it;
		for (it = charts.begin(); it != charts.end(); ) {
			if ((*it)->selected) {
				delete* it;
				it = charts.erase(it);
			}
			else {
				++it;
			}
		}
		resetSelectedChart();
	}
	ImGui::SliderFloat("Plots Shade Opacity", &opacity, 0.0f, 1.0f);
	ImGui::Checkbox("Anti-aliased lines", &style.AntiAliasedLines);

	// Dropdown to choose well unit
	if (ImGui::BeginCombo("Well unit", combo_preview_value, flags))
	{
		for (int n = 0; n < 3; n++)
		{
			const bool is_selected = (selectedWellUnit == WellUnit(n));
			if (ImGui::Selectable(getWellUnit(WellUnit(n)), is_selected)) {
				selectedWellUnit = WellUnit(n);
				trackRule->changeWellUnit(selectedWellUnit);
			}
			// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	colorPicker(&background_color, "Background Color");
	ImGui::ShowFontSelector("Font Selector");
	ImGui::SliderFloat("Line Thickness", &line_weight, 0.0f, 3.0f, "ratio = %.2f");

	ImGui::DragFloat("Track width", &chart_width, 5.0f, 200.0f, 500.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);

	if (ImGui::DragFloat("Range limit", &r_limit, 10.0f, 200.0f, 2000.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
		r_change = true;
	}

	if (ImGui::TreeNode("Filling")) {
		for (int i = 0; i < charts.size(); i++) {
			ImGui::DragFloat(("Chart " + std::to_string(i + 1)).c_str(), &charts[i]->fill_line, 1.0f, charts[i]->attr_min, charts[i]->attr_max, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		}
		ImGui::TreePop();
	}
}
void PlotWithMultipleKeys::markerSetting() {
	if (ImGui::TreeNode("Markers")) {
		if (!m_picks.isEmpty()) {
			for (int i = 0; i < m_picks.size(); i++) {

				if (ImGui::Button((m_picks.at(i)->markerName().toStdString() + " " + m_picks[i]->kind().toStdString()).c_str())) {
					r_min = m_picks.at(i)->value() - r_limit / 2;
					r_max = m_picks.at(i)->value() + r_limit / 2;
					r_change = true;
				}
			}
		}
		ImGui::TreePop();
	}
}
void colorPicker(ImVec4* tagColor, std::string lname) {
	bool saved_palette_init = true;
	ImVec4 color = *tagColor;
	ImVec4 saved_palette[32] = { };
	if (saved_palette_init) {
		for (int n = 0; n < IM_ARRAYSIZE(saved_palette); n++) {
			ImGui::ColorConvertHSVtoRGB(n / 31.0f, 0.8f, 0.8f,
				saved_palette[n].x, saved_palette[n].y,
				saved_palette[n].z);
			saved_palette[n].w = 1.0f; // Alpha
		}
		saved_palette_init = false;
	}
	ImVec4 backup_color;
	bool open_popup = ImGui::ColorButton(("MyColor##3b" + lname).c_str(), color);
	ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
	open_popup |= ImGui::Button(("Pick Your Color##" + lname).c_str());
	if (open_popup) {
		ImGui::OpenPopup(("mypicker" + lname).c_str());
		backup_color = color;
	}
	if (ImGui::BeginPopup(("mypicker" + lname).c_str())) {
		ImGui::Text("MY CUSTOM COLOR PICKER WITH AN AMAZING PALETTE!");
		ImGui::Separator();
		ImGui::ColorPicker4(("##picker" + lname).c_str(), (float*)&color,
			ImGuiColorEditFlags_NoSidePreview
			| ImGuiColorEditFlags_NoSmallPreview);
		ImGui::SameLine();

		ImGui::BeginGroup(); // Lock X position
		ImGui::Text("Current");
		ImGui::ColorButton(("##current" + lname).c_str(), color,
			ImGuiColorEditFlags_NoPicker
			| ImGuiColorEditFlags_AlphaPreviewHalf,
			ImVec2(60, 40));
		ImGui::Text("Previous");
		if (ImGui::ColorButton(("##previous" + lname).c_str(), backup_color,
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
			if (ImGui::ColorButton(("##palette" + lname).c_str(), saved_palette[n],
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
		if (ImGui::Button(("Close ##" + lname).c_str())) {
			ImGui::CloseCurrentPopup();
		}
		*tagColor = color;
		ImGui::EndGroup();
		ImGui::EndPopup();
	}
}
void PlotWithMultipleKeys::chartHeader() {
	if (selectedWell == -1)
		ImGui::Text("No Well Bore has been selected");
	else {
		ImGui::Text(("Well Bore name: " + bore->name().toStdString()).c_str());
	}
}
void PlotWithMultipleKeys::showActiveLog(log_data* plog) {
	ImDrawList* draw_list = ImGui::GetWindowDrawList();
	std::string lname = plog->logView->log_name.toStdString() + "##" + plog->wellbore->name().toStdString();
	//popup
	if (ImGui::Button(plog->logView->log_name.toStdString().c_str())) {
		ImGui::OpenPopup((lname).c_str());
	}
	if (ImGui::BeginPopup((lname).c_str())) {
		double min = plog->logView->acitiveChart->attr_min;
		double max = plog->logView->acitiveChart->attr_max;
		if (ImGui::Button(("Pop Log Out Of Chart##" + lname).c_str())) {
			removeLog(plog->logView->acitiveChart, plog);
			ImGui::CloseCurrentPopup();
		}
		// filling option

		ImGui::SliderFloat(("Line Thickness##" + lname).c_str(), &plog->thickness, 0.0f, 3.0f, "ratio = %.2f");
		ImGui::SameLine();
		ImGui::Checkbox(("Global thickness ##" + plog->logView->log_name.toStdString()).c_str(), &plog->isGlobalThickness);
		ImGui::SliderFloat(("Fill opacity##" + lname).c_str(), &plog->opacity, 0.0f, 1.0f, "ratio = %.2f");
		ImGui::DragScalar(("Fill Value##" + plog->logView->log_name.toStdString()).c_str(),
			ImGuiDataType_Double, &plog->fillValue
			, (plog->attr_max_value - plog->attr_min_value) / 20.0f
			, &min, &max, "%0.3f");
		ImGui::SameLine();
		ImGui::Checkbox(("filled ##" + plog->logView->log_name.toStdString()).c_str(), &plog->shaded);
		//color picker
		colorPicker(&plog->logView->color, lname);
		if (ImGui::Button(("Close ##" + lname).c_str())) {
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
	//legend
	ImGui::Text("%0.2f", plog->attr_min_value);
	ImGui::SameLine(0, 4.0f);

	ImVec2 p = ImGui::GetCursorScreenPos();
	float th = 3.0f;
	float x = p.x;
	float y = p.y + 9.0f;
	float sz = ImGui::GetContentRegionAvail().x * 5.5f / 10.0f;

	draw_list->AddLine(ImVec2(x, y), ImVec2(x + sz, y), ImColor(plog->logView->color), th);

	ImGui::SameLine(0, sz * 1.2);
	ImGui::Text("%0.2f", plog->attr_max_value);
	ImGui::Text(("Unit: " + plog->sUnit).c_str());

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

	window->DrawList->AddLine(ImVec2(0, mousePos.y + viewport->WorkPos.y),
		ImVec2(viewport->WorkSize.x, mousePos.y + viewport->WorkPos.y),
		ImGui::GetColorU32(ImVec4(1, 0, 0, 255)), 1.0f);

	window->DrawList->AddLine(ImVec2(mousePos.x + viewport->WorkPos.x, 0),
		ImVec2(mousePos.x + viewport->WorkPos.x, viewport->WorkSize.y),
		ImGui::GetColorU32(ImVec4(1, 0, 0, 255)), 1.0f);
}

void  PlotWithMultipleKeys::widgetStyle() {
	ImGui::StyleColorsLight();

	ImGuiStyle& style = ImGui::GetStyle();
	style.Colors[ImGuiCol_WindowBg] = background_color;
	style.Colors[ImGuiCol_PopupBg] = background_color;
}

bool PlotWithMultipleKeys::showDepth() {
	ImGui::BeginGroup();
	ImGui::BeginChild("##Legend bar chart", ImVec2(chart_width / 2.0f, ImGui::GetContentRegionAvail().y / 5), true);
	ImGui::EndChild();

	ImPlot::SetNextAxisLinks(ImAxis_Y1, &lims->Y.Min, &lims->Y.Max);

	if (ImPlot::BeginPlot("##chart Rule", ImVec2(chart_width / 2.0f, -1),
		ImPlotFlags_WheelUnchangedRange | ImPlotFlags_NoCanvas)) {
		ImPlot::SetupAxis(ImAxis_X1, NULL, ImPlotAxisFlags_Opposite | ImPlotAxisFlags_NoDecorations);

		ImPlot::SetupAxis(ImAxis_Y1, getWellUnit(selectedWellUnit), ImPlotAxisFlags_Invert);
		if (!m_picks.isEmpty()) {

			for (int i = 0; i < m_picks.size(); ++i) {

				float red = ((float)m_picks.at(i)->currentMarker()->color().red()) / 255;
				float green = ((float)m_picks.at(i)->currentMarker()->color().green()) / 255;
				float blue = ((float)m_picks.at(i)->currentMarker()->color().blue()) / 255;

				ImPlot::TagY(m_picks.at(i)->value(), ImVec4(red, green, blue, 1.0f),
					(m_picks.at(i)->markerName().toStdString() + m_picks.at(i)->kind().toStdString()).c_str());

			}
		}

		// Show the long crosshair cursor
		if (useLongCrossHair) {
			longCrossHairCursor();
		}
		//change chart limit value
	//	if (linkDepthAxis) {
		ImPlotRect rect = ImPlot::GetPlotLimits();
		//	}
		ImPlot::EndPlot();
	}
	ImGui::EndGroup();
	ImGui::SameLine();
}
void PlotWithMultipleKeys::resetSelectedChart() {
	for (int i = 0; i < charts.size(); i++) {
		charts[i]->selected = false;
	}
};
// Drag and drop to plot a log
void PlotWithMultipleKeys::showPlot() {
	widgetStyle();
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
		ImGui::BeginChild("MENU_BAR", ImVec2(-1, 100), true, winflags);
		ImGuiStyle& style = ImGui::GetStyle();
		menubar(style);
		ImGui::EndChild();
		// child window to serve as initial source for our DND items
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
		ImGui::BeginChild("DND_LEFT", ImVec2(ImGui::GetContentRegionAvail().x * 0.2f, -1), false, window_flags);

		// Set mouse cursor
		if (ImGui::IsWindowHovered())
			ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

		// Define arrays storing depth, log and seismic
		int numLogs = 0; // intialization of numLogs here is mandatory
		int numSeismics = listSeismicDatasets.size();

		std::vector<std::string> logNames;
		std::vector<std::string> seismicNames;
		std::vector<double> seismic_extract;
		std::vector<double> depth_seismic;
		static const double dDepth_seismic = 0.2; //we consider a step of 0.2 meter per seismic points.
		static int numPoints_seismic;

		//std::string logName = " ";
		//float const nullValue = -999.25;
		//float logMin = 1e6;
		//float logMax = 1e-6;
		//float depthMin = 1e6;
		//float depthMax = 1e-6;

		// Initialization
		WellBore* bore;

		//bool logIsSelected;
		//Logs currentLog;
		static WellUnit keyUnit = MD;
		// Checkbox to set linked depth axis
		if (ImGui::Button("Setting")) {
			ImGui::OpenPopup("Setting");
		}
		if (ImGui::BeginPopup("Setting")) {
			setting(style);
			ImGui::EndPopup();
		}
		// Render marker setting
		markerSetting();
		//		static int selectedWell = -1;
		if (selectedWell > -1)
		{
			bore = listWellBores[selectedWell];
			m_picks = bore->picks();
			numLogs = bore->logsNames().size();
		}
		else {
			bore = nullptr;
		}
		if (ImGui::TreeNode("WellBores"))
		{
			if (bore != nullptr) {
				for (int i = 0; i < numLogs; i++) {
					log_data* p_log = findLogViewByLogname(bore->logsNames()[i])->findByWellBore(bore);

					if (p_log->is_empty) {
						continue;
					}
					if (p_log->logView->acitiveChart == nullptr) {
						ImPlot::ItemIcon(p_log->logView->color);

						ImGui::SameLine();
						ImGui::Selectable(p_log->logView->log_name.toStdString().c_str(), false, 0, ImVec2(100, 0));
						if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
							ImGui::SetDragDropPayload("MY_DND", &i, sizeof(int));
							ImPlot::ItemIcon(p_log->logView->color); ImGui::SameLine();
							ImGui::TextUnformatted(p_log->logView->log_name.toStdString().c_str());
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
			ImGui::TreePop();
		}

		ImGui::EndChild();

		// Drag and Drop target
		if (ImGui::BeginDragDropTarget()) {
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MY_DND")) {
				int i = *(int*)payload->Data; processed_logs_ptr[i].logView->reset();

			}
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ADD_TRACK")) {

			}
			ImGui::EndDragDropTarget();
		}

		ImGui::SameLine();

		// Plot
		{
			ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysHorizontalScrollbar;

			ImGui::BeginChild("DND_RIGHT", ImVec2(-1, -1), true, window_flags);
			ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
			if (ImGui::BeginTabBar("##WellBoresTabBar", tab_bar_flags))
			{
				for (int n = 0; n < totalNumberOfWellBores; n++)
				{
					if (ImGui::BeginTabItem(listWellBores[n]->name().toStdString().c_str()))
					{
						selectedWell = n;
						ImGui::EndTabItem();
					}
				}

				ImGui::EndTabBar();
			}
			chartHeader();

			int point_idx, seismic_point_idx;
			std::vector<int> logIdx;
			std::vector<int> seismicIdx;
			//	static ImPlotRect lims(0, 1, 0, 2000);
			ImRect inner_rect = ImGui::GetCurrentWindow()->InnerRect;

			for (int col = 0; col < charts.size(); col++)
			{
				int pltIdx = col;
				/*if (col == trackRule->column) {
					showDepth();
					continue;
				}
				else if (col > trackRule->column) {
					pltIdx = col - 1;
				}*/

				charts[pltIdx]->currentBore(bore);
				std::string label = std::to_string(pltIdx);


				ImGui::BeginGroup();

				// Drag and Drop target
				ImGui::BeginChild(("##Legend bar" + label).c_str(), ImVec2(chart_width, ImGui::GetContentRegionAvail().y / 5), true);

				for (int i = 0; i < charts[pltIdx]->v_listLogs.size(); i++) {
					log_data* p_log = charts[pltIdx]->v_listLogs[i]->findByWellBore(bore);
					if (p_log == nullptr)
						continue;
					showActiveLog(p_log);

				}
				ImGui::EndChild();

				// Plot
				if (r_change) {

					if (pltIdx == charts.size() - 1) {
						r_change = false;
					}
					lims->Y.Max = lims->Y.Min + r_limit;
					ImPlot::SetNextAxisLimits(ImAxis_Y1, r_min, r_max, ImPlotCond_Always);

				}
				if (!charts[pltIdx]->autofit)
					ImPlot::SetNextAxisLinks(ImAxis_X1, &charts[pltIdx]->x_min, &charts[pltIdx]->x_max);
				if (linkDepthAxis[selectedWell]) {
					ImPlot::SetNextAxisLinks(ImAxis_Y1, &lims->Y.Min, &lims->Y.Max);
				}
				else {
					ImPlot::SetNextAxisLinks(ImAxis_Y1, &charts[pltIdx]->y_min, &charts[pltIdx]->y_max);
				}

				if (ImPlot::BeginPlot(("##DND" + std::to_string(pltIdx)).c_str(), ImVec2(chart_width, -1),
					ImPlotFlags_WheelUnchangedRange | ImPlotFlags_NoLegend)) {
					if (ImGui::IsItemActive()) {
						if (ImGui::GetIO().KeyCtrl)
						{
							charts[pltIdx]->selected = true;
						}
						else {
							this->resetSelectedChart();
							charts[pltIdx]->selected = true;
						}
					}
					if (charts[pltIdx]->selected) {
						ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0.5, 0.5, 0.5, 0.9));
					}
					ImPlotAxisFlags xAxisFlag = charts[pltIdx]->autofit ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_Lock;
					ImPlot::SetupAxis(ImAxis_X1, NULL, ImPlotAxisFlags_Opposite | ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoTickLabels | xAxisFlag);


					/*if (pltIdx == 0) {
						ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_Opposite | ImPlotAxisFlags_Invert);
						if (!m_picks.isEmpty()) {

							for (int i = 0; i < m_picks.size(); ++i) {

								float red = ((float)m_picks.at(i)->currentMarker()->color().red()) / 255;
								float green = ((float)m_picks.at(i)->currentMarker()->color().green()) / 255;
								float blue = ((float)m_picks.at(i)->currentMarker()->color().blue()) / 255;

								ImPlot::TagY(m_picks.at(i)->value(), ImVec4(red, green, blue, 1.0f),
									(m_picks.at(i)->markerName().toStdString() + m_picks.at(i)->kind().toStdString()).c_str());

							}
						}
					}
					else*/
					//		ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_Invert| ImPlotAxisFlags_NoDecorations);

					if (linkDepthAxis[selectedWell])
					{
						ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_Invert | ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoTickLabels);
					}
					else {
						ImPlot::SetupAxis(ImAxis_Y1, NULL, ImPlotAxisFlags_Invert);
					}
					for (int i = 0; i < charts[pltIdx]->v_listLogs.size(); i++) {
						log_data* p_log = charts[pltIdx]->v_listLogs[i]->findByWellBore(bore);
						if (p_log == nullptr)
							continue;
						int count = p_log->end - p_log->start;
						if (!p_log->is_init) {
							p_log->initialize_data();
						}
						if (p_log->cur_unit != selectedWellUnit) {
							p_log->update_keys_on_unit(selectedWellUnit);
						}
						ImPlot::SetNextLineStyle(p_log->logView->color);
						ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, p_log->isGlobalThickness ? line_weight : p_log->thickness);
						ImPlot::PlotLine(p_log->logView->log_name.toStdString().c_str(),
							p_log->attributes,
							p_log->keys, p_log->num_points);
						ImPlot::PopStyleVar();
						point_idx = interactiveHelper(p_log->keys);
						if (p_log->shaded) {
							ImVec4 v4col = p_log->logView->color;
							v4col.w = p_log->opacity;
							ImU32 col = ImGui::GetColorU32(v4col);
							ImPlot::PlotShadedV(p_log->logView->log_name.toStdString().c_str(),
								p_log->attributes,
								p_log->fillValue,
								p_log->keys,
								p_log->num_points, col);

						}
					}
					// allow the main plot area to be a DND target
					if (ImPlot::BeginDragDropTargetPlot()) {
						if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MY_DND")) {
							int i = *(int*)payload->Data;
							QString lname = bore->logsNames()[i];
							// charts[pltIdx].add_log(&processed_logs_ptr[i]);
							addLogInChart(pltIdx, lname);
						}
						if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ADD_TRACK")) {
							std::cout << "Add track" << std::endl;
							chart* newChart = new chart();
							charts.insert(charts.begin() + pltIdx, newChart);
						}
						ImPlot::EndDragDropTarget();
					}
					// Show the long crosshair cursor
					if (useLongCrossHair) {
						longCrossHairCursor();
					}
					//change chart limit value
				//	if (linkDepthAxis) {
			//		ImPlotRect rect = ImPlot::GetPlotLimits();

					//charts[pltIdx]->chartLimitValues(&rect.X.Min, &rect.X.Max, &rect.Y.Min, &rect.Y.Max);

					//	}
					if (charts[pltIdx]->selected) {
						ImPlot::PopStyleColor();
					}
					ImPlot::EndPlot();
				}
				ImGui::EndGroup();
				ImGui::SameLine();
			}
			if (ImGui::BeginDragDropTargetCustom(inner_rect, ImGui::GetID("all bg"))) {
				if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ADD_TRACK")) {
					std::cout << "Add track" << std::endl;
					if (charts.size() == 0) {
						chart* newChart = new chart();
						charts.push_back(newChart);
					}
				}
				ImGui::EndDragDropTarget();
			}
			if (useLongCrossHair) {
				longCrossHairCursor();
			}
			ImGui::EndChild();
		}
		ImGui::End();
	}
}


