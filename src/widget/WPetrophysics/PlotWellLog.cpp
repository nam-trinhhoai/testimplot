/*
 *
 *
 *  Created on: 15 June 2022
 *      Author: l0359127
 */

#include "PlotWellLog.h"

PlotWellLog::PlotWellLog(WorkingSetManager* manager):
m_manager(manager)
{
/**
	WorkingSetManager::FolderList folders = m_manager->folders();
	FolderData* wells = folders.wells;


	QList<IData*> iData = wells->data();
	

	WellHead *wellHead = dynamic_cast<WellHead*>(iData[0]);


	WellBore *bore = wellHead->wellBores()[0];


	std::vector<std::string> logNames;
	for (int i=0; i<bore->logsNames().size();i++)
	{
		logNames.push_back( bore->logsNames()[i].toStdString() );	
	}


	int logIdx = 0;
	bool logIsSelected = bore->selectLog(logIdx);

	Logs myLog = bore->currentLog();

	numPoints = myLog.attributes.size();
	
	double *depth = &myLog.keys[0];
	double *log = &myLog.attributes[0];

*/
}

PlotWellLog::~PlotWellLog()
{}

// Main code
void PlotWellLog::showPlot()
{
	// Create the main menu
	mainMenu();
	
	// Create the main layout
	//mainLayout();

	// Show the windows based on user command
	if (showLASContents)
		showLASContentsDialog();
	if (showQuickPlot_DB)
		showWellLogs_QuickPlot_DB();	
	if (showLogViewer)
		showWellLogs_Subplots();
	if (showLogViewer_table)
		showWellLogs_Tables();	
	if (showLogViewer_dragAndDrop)
		showWellLogs_dragAndDrop();		
	//if (showImPlotDemo)
	//	ImPlot::ShowDemoWindow(&showImPlotDemo);
	//if (showImGuiDemo)
	//	ImGui::ShowDemoWindow(&showImGuiDemo);
	if (showPlotStyle)
		showPlotStyleDialog();

}

// Main menu
void PlotWellLog::mainMenu()
{
	bool importLASFile = false;
	
    if(ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Data"))
        {
            if (ImGui::MenuItem("Load LAS input", NULL))
			{
                importLASFile = true;
            }
			
		    ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Visualization"))
        {
            if (ImGui::MenuItem("Show LAS contents", NULL))
                showLASContents = true;

			if (ImGui::MenuItem("Quick plot DB", NULL))
				showQuickPlot_DB = true;

			if (ImGui::MenuItem("Well Log Viewer", NULL))
				showLogViewer = true;

			if (ImGui::MenuItem("Well Log Viewer_Table", NULL))
				showLogViewer_table = true;

			if (ImGui::MenuItem("Well Log Viewer_Drag and Drop", NULL))
				showLogViewer_dragAndDrop = true;

     		if (ImGui::MenuItem("ImGui Demo", NULL))
				showImGuiDemo = true;
			
			if (ImGui::MenuItem("ImPlot Demo", NULL))
				showImPlotDemo = true;

			if (ImGui::MenuItem("Plot style", NULL))
				showPlotStyle = true;
			
		    ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
    
    // Read LAS file
	std::string const popupTitle = "Import LAS File"; 
	ImVec2 popupSize = ImVec2(600, 400);

    if(importLASFile)
        ImGui::OpenPopup(popupTitle.c_str());      
		
    if(file_dialog.showFileDialog(popupTitle.c_str(), imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, popupSize, ".las"))
    {
		// Get LAS file name
		LASFileName = file_dialog.selected_path;

		WellLogsIO *wellLogsIO;
		// Read header of a LAS file and store the log names in a vector
		bool readHeader = wellLogsIO->readLASHeader(LASFileName, logNames);

		// Get data from a LAS file and store in a vector
		bool readData = wellLogsIO->readLASData(LASFileName, logData, numPoints);	

		showLASContents = true;

		// Multiwell data
		LASFileName_multiWells.push_back(LASFileName);
		numPoints_multiWells.push_back(numPoints);		
		logNames_multiWells.push_back(logNames);
		logData_multiWells.push_back(logData);
    }
}

// Create main layout
void PlotWellLog::mainLayout()
{
	ImGuiViewport const *main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(main_viewport->WorkPos.x, main_viewport->WorkPos.y));
    ImGui::SetNextWindowSize(main_viewport->WorkSize);

	ImGui::Begin(" ");
	ImGui::End();
}

/*
 * Setup window size and position
 * Cloned from imgui demo
 */
void PlotWellLog::setWindowSize()
{
	ImGuiViewport const *main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(main_viewport->WorkPos.x+10, main_viewport->WorkPos.y+10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
}

void PlotWellLog::showLASContentsDialog()
{
	setWindowSize();
	
	ImGui::Begin("LAS file contents", &showLASContents, ImGuiWindowFlags_MenuBar);

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::BeginMenu("Options"))
		{
			showFileContentFlag = -1;
			showQuickPlot = false;			
		
		    if (ImGui::MenuItem("Loaded file", NULL))  
				showFileContentFlag = 1;
			if (ImGui::MenuItem("Data size"))
				showFileContentFlag = 0;
			if (ImGui::MenuItem("Curve names", NULL))
				showFileContentFlag = 2;
			if (ImGui::MenuItem("Loaded Data", NULL))
				showFileContentFlag = 3;
			if (ImGui::MenuItem("Quick plot", NULL))
				showFileContentFlag = 10;

		    ImGui::EndMenu();
		}
		ImGui::EndMenuBar();
	}	

	switch(showFileContentFlag)
	{
		case 0:
		{
			ImGui::BeginChild("Scrolling");
			
			ImGui::Text(("Number of logs:" + std::to_string(logNames_multiWells.back().size())).c_str(), 0);
			ImGui::Text(("Number of depth points:" + std::to_string(numPoints)).c_str(), 1);
			ImGui::Text(("Number of values:" + std::to_string(logData.size())).c_str(), 2);

			ImGui::EndChild();
			break;
		}
		case 1:
		{
			ImGui::BeginChild("Scrolling");

			ImGui::Text(file_dialog.selected_path.c_str(),0);

			ImGui::EndChild();
			break;
		}
		case 2:
		{
			ImGui::BeginChild("Scrolling");

			for(std::string & line : logNames_multiWells.back())
			    ImGui::Text(line.c_str());

			ImGui::EndChild();
			break;
		}
		case 3:
		{
			ImGui::BeginChild("Scrolling");

			int numLogs = logNames_multiWells.back().size();
			for(int i=0; i<numLogs; i++)
			{
				std::string logName = logNames_multiWells.back()[i];

		    	ImGui::Text((logName+"\t").c_str());
				ImGui::SameLine();
			}
			ImGui::NewLine();

			for(int pointIdx=0; pointIdx<numPoints; pointIdx++)
			{				
				for(int i=0; i<numLogs; i++)
				{
					std::string value = logData[pointIdx*numLogs+i];
			    	ImGui::Text(value.c_str());
					ImGui::SameLine();
				}
				ImGui::NewLine();
			}
			ImGui::EndChild();
			break;
		}
		case 10:
		{
			showQuickPlot = true;				
			break;
		}
	}

	if (showQuickPlot)
		showWellLogs_QuickPlot();

	ImGui::End();
}

// Plot the well logs in a table format
void PlotWellLog::showWellLogs_Tables() {

	if(logNames_multiWells.back().size()>0)
	{
#ifdef IMGUI_HAS_TABLE
		static ImGuiTableFlags flags = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV |
		                               ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable;
		
		ImGui::Begin("Well Log Viewer (Table)", &showLogViewer_table);

		if (ImGui::BeginTable("##table", 3, flags, ImVec2(-1,-1))) {

			// Header
			ImGui::TableSetupColumn("Sonic", ImGuiTableColumnFlags_WidthFixed, 75.0f);
			ImGui::TableSetupColumn("Gamma ray", ImGuiTableColumnFlags_WidthFixed, 75.0f);
			ImGui::TableSetupColumn("Density", ImGuiTableColumnFlags_WidthFixed, 75.0f);
			ImGui::TableHeadersRow();
			ImPlot::PushColormap(ImPlotColormap_Cool);

			static ImPlotRect lims(0,1,10000,0);

		    ImPlot::PushColormap(ImPlotColormap_Cool);

			int listLogToPlot[3] = {3, 9, 15};
			float logMin[3] = {40, 0, 2.0};
			float logMax[3] = {120, 100, 2.8};
		
	/**
			ImGui::TableNextRow();

			// Show log limits following petro log standard        
			//ImGui::TableSetColumnIndex(0);		
			//showLogLimits("40", "120");

			ImGui::TableSetColumnIndex(1);		
			showLogLimits("0", "100");

			ImGui::TableSetColumnIndex(2);		
			showLogLimits("2.0", "2.8");
	*/

			ImGui::TableNextRow();
		    for (int row = 0; row < 3; row++) {
		        //ImGui::TableNextRow();

		        float depth[numPoints];
				float log[numPoints];
	
				int numLogs = logNames_multiWells.back().size();

				for(int i=0; i<numPoints; i++)
				{
					depth[i] = std::stof(logData[numLogs*i + 0]);
					log[i] = std::stof(logData[numLogs*i + listLogToPlot[row]]);
				}

		        ImGui::TableSetColumnIndex(row);
		        ImGui::PushID(row);
		        
			
				if (ImPlot::BeginPlot("", ImVec2(-1,-1))) // Added ImVec2 to set chart size
				{	if (row == 0)
						ImPlot::SetupAxis(ImAxis_Y1,NULL, ImPlotAxisFlags_Invert);	
					if (row > 0)
						ImPlot::SetupAxis(ImAxis_Y1,NULL,ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_Invert);

					ImPlot::SetupAxesLimits(logMin[row], logMax[row], 1, 0);
					ImPlot::SetupAxisLinks(ImAxis_Y1, &lims.Y.Max, &lims.Y.Min);

					// Set line color
					ImPlot::SetNextLineStyle(ImPlot::SampleColormap(row,ImPlotColormap_Cool));
		            
					ImPlot::PlotLine(logNames_multiWells.back()[listLogToPlot[row]].c_str(), log, depth, numPoints);
					ImPlot::EndPlot();
				}
		
		
		        ImGui::PopID();
		    }
		    ImPlot::PopColormap();
		    ImGui::EndTable();
		}

		ImGui::End();
	
#else
    	ImGui::BulletText("You need to merge the ImGui 'tables' branch for this section.");
#endif
	}
	else
	{
		ImGui::Begin("Warning", &showLogViewer_table);
		ImGui::Text("Input LAS file is missing!");
		ImGui::End();
	}
}

/**
void PlotWellLog::showLogLimits(std::string const logMin, std::string const logMax)
{

	ImGui::Text(logMin.c_str());
	ImGui::SameLine();
	ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetColumnWidth() - ImGui::CalcTextSize(logMax.c_str()).x 
		- ImGui::GetScrollX() - 2 * ImGui::GetStyle().ItemSpacing.x);
	ImGui::Text(logMax.c_str());
}
*/


void PlotWellLog::showPlotStyleDialog() {

	ImGui::Begin("Plot style", &showPlotStyle);

    ImPlot::ShowStyleSelector("Style");
    ImPlot::ShowColormapSelector("Colormap");
    ImGui::Checkbox("Anti-Aliased Lines", &ImPlot::GetStyle().AntiAliasedLines);

 	ImGui::End();
}



// Plot the well logs in quickplot format
void PlotWellLog::showWellLogs_QuickPlot() {

	ImGui::Begin("Quick plot");

	// Get data
	float depth[numPoints];
	float log[numPoints];
	std::string logName = " ";
	float const nullValue = -999.25;
	float logMin = 1e6;
	float logMax = 1e-6;
	float depthMin = 1e6;
	float depthMax = 1e-6;

	int numLogs = logNames_multiWells.back().size();

	// Get the depth
	for(int i=0; i<numPoints; i++)
	{
		depth[i] = std::stof(logData[numLogs*i + 0]);
		
		// Update depth range
		if(depth[i] != nullValue)
		{
			if(depth[i] < depthMin)
				depthMin = depth[i];
			if(depth[i] > depthMax)
				depthMax = depth[i];
		}
	}	

	// List of logs
	{
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
		        
		ImGui::BeginChild("ChildLeft", ImVec2(ImGui::GetContentRegionAvail().x*0.2f, -1), false, window_flags);

		if (ImGui::TreeNode("Logs"))
		{
		    static int selected = -1;
		    for (int n = 0; n < logNames_multiWells.back().size(); n++)
		    {
		        if (ImGui::Selectable(logNames_multiWells.back()[n].c_str(), selected == n))
		            selected = n;
		    }
		    ImGui::TreePop();

			if(selected >-1)
			{
				logName = logNames_multiWells.back()[selected];

				for(int i=0; i<numPoints; i++)
				{
					log[i] = std::stof(logData[numLogs*i + selected]);
					
					// Update data range
					if(log[i] != nullValue)
					{
						if(log[i] < logMin)
							logMin = log[i];
						if(log[i] > logMax)
							logMax = log[i];
					}
				}
			}
		}

		ImGui::EndChild();
	}
	ImGui::SameLine();

	// Plot
	{
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
		
		//ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
		ImGui::BeginChild("ChildRight", ImVec2(-1, -1), true, window_flags);
		if (ImPlot::BeginPlot(logName.c_str(), ImVec2(-1,-1))) 
		{
			ImPlot::SetupAxesLimits(logMin, logMax, depthMin, depthMax);
			ImPlot::SetupAxis(ImAxis_X1,logName.c_str());
			ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);	
			ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);
			ImPlot::EndPlot();
		}
		ImGui::EndChild();
	}

	ImGui::End();


}


// Plot the well logs from database in quickplot format
void PlotWellLog::showWellLogs_QuickPlot_DB() {
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
		ImGui::Begin("Quick plot DB");
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
		ImGui::BeginChild("ChildLeft", ImVec2(ImGui::GetContentRegionAvail().x*0.2f, -1), false, window_flags);


		// Get data
		double *depth;
		double *log;

		std::string logName = " ";
		float const nullValue = -999.25;
		float logMin = 1e6;
		float logMax = 1e-6;
		float depthMin = 1e6;
		float depthMax = 1e-6;

		// Initialization
		//WellHead *wellHead = dynamic_cast<WellHead*>(iData[0]);
		//WellBore *bore = wellHead->wellBores()[0];

		WellBore *bore = listWellBores[0];
		bool logIsSelected = bore->selectLog(0);
		Logs currentLog = bore->currentLog();
		numPoints = currentLog.attributes.size();
		depth = &currentLog.keys[0];
		log = &currentLog.attributes[0];


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

				int logIdx = 0;
				bool logIsSelected = bore->selectLog(logIdx);

				Logs currentLog = bore->currentLog();

				numPoints = currentLog.attributes.size();


				depth = &currentLog.keys[0];
				log = &currentLog.attributes[0];

				int numLogs = bore->logsNames().size();

				// Get the depth
				for(int i=0; i<numPoints; i++)
				{
					//depth[i] = std::stof(logData[numLogs*i + 0]);
	
					// Update depth range
					if(depth[i] != nullValue)
					{
						if(depth[i] < depthMin)
							depthMin = depth[i];
						if(depth[i] > depthMax)
							depthMax = depth[i];
					}
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
							//logName = logNames_multiWells.back()[selected];
							logName = bore->logsNames()[selected].toStdString();

							logIsSelected = bore->selectLog(selected);

							currentLog = bore->currentLog();
							numPoints = currentLog.attributes.size();

							log = &currentLog.attributes[0];

							for(int i=0; i<numPoints; i++)
							{
								//log[i] = std::stof(logData[numLogs*i + selected]);
				
								// Update data range
								if(log[i] != nullValue)
								{
									if(log[i] < logMin)
										logMin = log[i];
									if(log[i] > logMax)
										logMax = log[i];
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
			if (ImPlot::BeginPlot(logName.c_str(), ImVec2(-1,-1))) 
			{
				ImPlot::SetupAxesLimits(logMin, logMax, depthMin, depthMax);
				ImPlot::SetupAxis(ImAxis_X1,logName.c_str());
				ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);	
				ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);
				ImPlot::EndPlot();
			}
			ImGui::EndChild();
		}

		ImGui::End();
	}
}


/**
// Plot the well logs in quickplot format
void PlotWellLog::showWellLogs_QuickPlot_DB() {

	WorkingSetManager::FolderList folders = m_manager->folders();
	FolderData* wells = folders.wells;


	QList<IData*> iData = wells->data();
	

	WellHead *wellHead = dynamic_cast<WellHead*>(iData[0]);


	WellBore *bore = wellHead->wellBores()[0];


	std::vector<std::string> logNames;
	for (int i=0; i<bore->logsNames().size();i++)
	{
		logNames.push_back( bore->logsNames()[i].toStdString() );	
	}


	int logIdx = 0;
	bool logIsSelected = bore->selectLog(logIdx);

	Logs myLog = bore->currentLog();

	numPoints = myLog.attributes.size();
	
	double *depth = &myLog.keys[0];
	double *log = &myLog.attributes[0];


	ImGui::Begin("Quick plot", &showQuickPlot_DB);

	// Plot
	{
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
		
		//ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
		ImGui::BeginChild("ChildRight", ImVec2(-1, -1), true, window_flags);
		if (ImPlot::BeginPlot(" ", ImVec2(-1,-1))) 
		{
			ImPlot::SetupAxis(ImAxis_X1,"X");
			ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);	
			ImPlot::PlotLine("logName", log, depth, numPoints);
			ImPlot::EndPlot();
		}
		ImGui::EndChild();
	}

	ImGui::End();

}
*/

// Plot the well logs in subplots format
void PlotWellLog::showWellLogs_Subplots() {
	if(logNames_multiWells.back().size()>0)
	{
		ImGui::Begin("Well log viewer - subplots", &showLogViewer);

		if (ImPlot::BeginSubplots("Well Logs", 1, 3, ImVec2(-1,-1), ImPlotSubplotFlags_NoTitle)) {
			
			static ImPlotRect lims(0,1,10000,0);

			int listLogToPlot[3] = {3, 9, 15};
			float logMin[3] = {40, 0, 2.0};
			float logMax[3] = {120, 100, 2.8};
	

			for (int row = 0; row < 3; row++) {
			    float depth[numPoints];
				float log[numPoints];

				int numLogs = logNames_multiWells.back().size();

				for(int i=0; i<numPoints; i++)
				{
					depth[i] = std::stof(logData[numLogs*i + 0]);
					log[i] = std::stof(logData[numLogs*i + listLogToPlot[row]]);
				}
			    
		
				if (ImPlot::BeginPlot("", ImVec2(-1,-1))) // Added ImVec2 to set chart size
				{	if (row == 0)
						ImPlot::SetupAxis(ImAxis_Y1,NULL, ImPlotAxisFlags_Invert);	
					if (row > 0)
						ImPlot::SetupAxis(ImAxis_Y1,NULL,ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_Invert);

			
					ImPlot::SetupAxesLimits(logMin[row], logMax[row], 1, 0);
					ImPlot::SetupAxisLinks(ImAxis_Y1, &lims.Y.Max, &lims.Y.Min);
					
					// Set line color
					ImPlot::SetNextLineStyle(ImPlot::SampleColormap(row,ImPlotColormap_Cool));
			        
					ImPlot::PlotLine(logNames_multiWells.back()[listLogToPlot[row]].c_str(), log, depth, numPoints);
					ImPlot::EndPlot();
				}
	
	
			}
			ImPlot::EndSubplots();
		}

		ImGui::End();
	}
	else
	{
		ImGui::Begin("Warning", &showLogViewer);
		ImGui::Text("Input LAS file is missing!");
		ImGui::End();
	}
}



// Drag and drop to plot a log
template <typename T>
inline T RandomRange(T min, T max) {
    T scale = rand() / (T) RAND_MAX;
    return min + scale * ( max - min );
}

ImVec4 RandomColor() {
    ImVec4 col;
    col.x = RandomRange(0.0f,1.0f);
    col.y = RandomRange(0.0f,1.0f);
    col.z = RandomRange(0.0f,1.0f);
    col.w = 1.0f;
    return col;
}


// Show the long crosshair cursor			
void longCrossHairCursor()
{
/*	int x, y;
	uint32 buttons;

	SDL_PumpEvents();  // make sure we have the latest mouse state.

	buttons = SDL_GetMouseState(&x, &y);

	SDL_DisplayMode DM;
	SDL_GetCurrentDisplayMode(0, &DM);

	ImGuiWindow* window = ImGui::GetCurrentWindow();

	window->DrawList->AddLine(ImVec2(0, y),ImVec2(DM.w, y), ImGui::GetColorU32(ImVec4(255, 255, 255, SDL_ALPHA_OPAQUE)), 1.0f);
	window->DrawList->AddLine(ImVec2(x, 0),ImVec2(x,DM.h), ImGui::GetColorU32(ImVec4(255, 255, 255, SDL_ALPHA_OPAQUE)), 1.0f);
*/
}

// convenience struct to manage DND items; do this however you like
struct MyDndItem {
    int              Idx;
    int              Plt;
	int chartIdx;
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



void PlotWellLog::showWellLogs_dragAndDrop() {
	if(logNames_multiWells.back().size()>0)
	{

		static bool use_work_area = true;
		static ImGuiWindowFlags winflags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;

		// We demonstrate using the full viewport area or the work area (without menu-bars, task-bars etc.)
		// Based on your use case you may want one of the other.
		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(use_work_area ? viewport->WorkPos : viewport->Pos);
		ImGui::SetNextWindowSize(use_work_area ? viewport->WorkSize : viewport->Size);		
		ImGui::Begin("View Well Logs_Drag and Drop", &showLogViewer_dragAndDrop, winflags);
		ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);


		// Get data
		float depth[numPoints];
		float log[numPoints];
		std::string logName = " ";
		float const nullValue = -999.25;
		float logMin = 1e6;
		float logMax = 1e-6;
		float depthMin = 1e6;
		float depthMax = 1e-6;

		int numLogs = logNames.size();

		// Get the depth
		for(int i=0; i<numPoints; i++)
		{
			depth[i] = std::stof(logData[numLogs*i + 0]);
		
			// Update depth range
			if(depth[i] != nullValue)
			{
				if(depth[i] < depthMin)
					depthMin = depth[i];
				if(depth[i] > depthMax)
					depthMax = depth[i];
			}
		}	
/**
		// convenience struct to manage DND items; do this however you like
		struct MyDndItem {
		    int              Idx;
		    int              Plt;
			int chartIdx;
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
*/
		// TODO set size of dnd=numLogs
		static MyDndItem dnd[100];

		

		// child window to serve as initial source for our DND items
		ImGui::BeginChild("DND_LEFT",ImVec2(100,-1));
		
		// Set mouse cursor
		if(ImGui::IsWindowHovered())		
			ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

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

		
		if (ImGui::TreeNode("Wells"))
		{
			static int selected = -1;
		    for (int n = 0; n < LASFileName_multiWells.size(); n++)
		    {
		        if (ImGui::Selectable(LASFileName_multiWells[n].c_str(), selected == n))
		            selected = n;
		    }
		    ImGui::TreePop();
/**
			if(selected >-1)			
			{
				logNames = logNames_multiWells[selected];
				logData = logData_multiWells[selected];
				numPoints = numPoints_multiWells[selected];
			}
*/
		}

		

		ImGui::EndChild();

		if (ImGui::BeginDragDropTarget()) {
		    if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MY_DND")) {
		        int i = *(int*)payload->Data; dnd[i].Reset();
		    }
		    ImGui::EndDragDropTarget();
		}

		ImGui::SameLine();
		ImGui::BeginChild("DND_RIGHT",ImVec2(-1,-1));

		ImPlotAxisFlags flags = ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines;

		// First plot
		if (ImPlot::BeginPlot("##DND1", ImVec2(300,-1))) {
		    ImPlot::SetupAxis(ImAxis_X1,NULL, ImPlotAxisFlags_Opposite);		
		    ImPlot::SetupAxis(ImAxis_Y1,"Depth", ImPlotAxisFlags_Invert);
			ImPlot::SetupAxesLimits(0, 100, depthMin, depthMax);

		    for (int k = 0; k < numLogs; ++k) {
		        if ((dnd[k].Plt == 1) & (dnd[k].chartIdx == 1)) {
		            //ImPlot::SetAxis(ImAxis_Y1);
		            ImPlot::SetNextLineStyle(dnd[k].Color);
		            
					logName = logNames[k];

					for(int i=0; i<numPoints; i++)
					{
						log[i] = std::stof(logData[numLogs*i + k]);
					/**
						// Update data range
						if(log[i] != nullValue)
						{
							if(log[i] < logMin)
								logMin = log[i];
							if(log[i] > logMax)
								logMax = log[i];
						}
					*/
					}

					ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);
		        }
		    }

		    // allow the main plot area to be a DND target
		    if (ImPlot::BeginDragDropTargetPlot()) {
		        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MY_DND")) {
		            int i = *(int*)payload->Data; dnd[i].Plt = 1; dnd[i].chartIdx = 1;
		        }
		        ImPlot::EndDragDropTarget();
		    }
		    
			// Show the long crosshair cursor	
			if (showLongCrossHairCursor)		
				longCrossHairCursor();

		    ImPlot::EndPlot();
		}
		
		// Second plot
		ImGui::SameLine();
		if (ImPlot::BeginPlot("##DND2", ImVec2(300,-1))) {
		    ImPlot::SetupAxis(ImAxis_X1,NULL, ImPlotAxisFlags_Opposite);		
		    ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);
			ImPlot::SetupAxesLimits(0, 100, depthMin, depthMax);

		    for (int k = 0; k < numLogs; ++k) {
		        if ((dnd[k].Plt == 1) & (dnd[k].chartIdx == 2)) {
		            //ImPlot::SetAxis(ImAxis_Y1);
		            ImPlot::SetNextLineStyle(dnd[k].Color);
		            
					logName = logNames[k];

					for(int i=0; i<numPoints; i++)
					{
						log[i] = std::stof(logData[numLogs*i + k]);
					}

					ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);
		        }
		    }

		    // allow the main plot area to be a DND target
		    if (ImPlot::BeginDragDropTargetPlot()) {
		        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MY_DND")) {
		            int i = *(int*)payload->Data; dnd[i].Plt = 1; dnd[i].chartIdx = 2;
		        }
		        ImPlot::EndDragDropTarget();
		    }
		    
			// Show the long crosshair cursor	
			if (showLongCrossHairCursor)		
				longCrossHairCursor();

		    ImPlot::EndPlot();
		}

		// Third plot
		ImGui::SameLine();
		if (ImPlot::BeginPlot("##DND3", ImVec2(300,-1))) {
		    ImPlot::SetupAxis(ImAxis_X1,NULL, ImPlotAxisFlags_Opposite);		
		    ImPlot::SetupAxis(ImAxis_Y1,"Depth",ImPlotAxisFlags_Invert);
			ImPlot::SetupAxesLimits(0, 100, depthMin, depthMax);

		    for (int k = 0; k < numLogs; ++k) {
		        if ((dnd[k].Plt == 1) & (dnd[k].chartIdx == 3)) {
		            //ImPlot::SetAxis(ImAxis_Y1);
		            ImPlot::SetNextLineStyle(dnd[k].Color);
		            
					logName = logNames[k];

					for(int i=0; i<numPoints; i++)
					{
						log[i] = std::stof(logData[numLogs*i + k]);
					}

					ImPlot::PlotLine(logName.c_str(), log, depth, numPoints);
		        }
		    }

		    // allow the main plot area to be a DND target
		    if (ImPlot::BeginDragDropTargetPlot()) {
		        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MY_DND")) {
		            int i = *(int*)payload->Data; dnd[i].Plt = 1; dnd[i].chartIdx = 3;
		        }
		        ImPlot::EndDragDropTarget();
		    }
		    
			// Show the long crosshair cursor	
			if (showLongCrossHairCursor)		
				longCrossHairCursor();

		    ImPlot::EndPlot();
		}

		// Show the long crosshair cursor	
		if (showLongCrossHairCursor)		
			longCrossHairCursor();
			
		ImGui::EndChild();

		ImGui::End();
	}
	else
	{
		ImGui::Begin("Warning", &showLogViewer_dragAndDrop);
		ImGui::Text("Input LAS file is missing!");
		ImGui::End();
	}
}

