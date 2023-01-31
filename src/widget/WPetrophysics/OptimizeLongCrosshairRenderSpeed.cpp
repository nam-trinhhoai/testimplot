/*
 *
 *
 *  Created on: 14 Sept 2022
 *      Author: l0359127
 */

#include "OptimizeLongCrosshairRenderSpeed.h"

OptimizeLongCrosshairRenderSpeed::OptimizeLongCrosshairRenderSpeed(WorkingSetManager* manager)
{}

OptimizeLongCrosshairRenderSpeed::~OptimizeLongCrosshairRenderSpeed()
{}


// Drag and drop to plot a log
void OptimizeLongCrosshairRenderSpeed::showPlot() {
	
	static bool use_work_area = true;
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);
	ImGui::SetNextWindowSize(use_work_area ? viewport->WorkSize : viewport->Size);

	bool showPlot = true;
	ImGui::Begin("#EmptyImGuiArea", &showPlot, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration);

	


	ImGuiWindow* window = ImGui::GetCurrentWindow();

	ImVec2 mousePos = ImGui::GetMousePos();
	window->DrawList->AddLine(ImVec2(0, mousePos.y),ImVec2(10000, mousePos.y), ImGui::GetColorU32(ImVec4(1, 0, 0, 255)), 1.0f);
	window->DrawList->AddLine(ImVec2(mousePos.x, 0),ImVec2(mousePos.x, 10000), ImGui::GetColorU32(ImVec4(1, 0, 0, 255)), 1.0f);

	ImGui::End();
}


