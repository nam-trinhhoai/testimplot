/*
 *
 *
 *  Created on: 07 Juin 2022
 *      Author: l0359127
 */

#include "PetroLogTools.h"

PetroLogTools::PetroLogTools()
{
    QWidget *widget = new QWidget;
    setCentralWidget(widget);

    //QWidget *topFiller = new QWidget;
    //topFiller->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    mainConsol = new QTextEdit();
    mainConsol->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);

	outputConsol = new QTextEdit();
	outputConsol->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
    
    //QWidget *bottomFiller = new QWidget;
    //bottomFiller->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    layout = new QHBoxLayout;
    layout->setContentsMargins(5, 5, 5, 5);
    layout->addWidget(mainConsol);
	layout->addWidget(outputConsol);
    widget->setLayout(layout);

    createActions();
    createMenus();

    QString message = tr("Tools for geomechanical simulation at well scale");
    statusBar()->showMessage(message);

    setWindowTitle(tr("Geomechanics at Well scale"));
    setMinimumSize(160, 160);
    resize(1012, 683);
}

#ifndef QT_NO_CONTEXTMENU
void PetroLogTools::contextMenuEvent(QContextMenuEvent *event)
{
    QMenu menu(this);
	menu.addAction(importLASAct);
    menu.addAction(plotWellLogsAct);
	menu.exec(event->globalPos());
}
#endif // QT_NO_CONTEXTMENU

void PetroLogTools::importLAS()
{
	QString fileName = QFileDialog::getOpenFileName(this,
    	tr("Open LAS file"), recentFileLocation.c_str(), tr("LAS input (*.las)"));
	
	load_file_to_qtextedit(fileName, mainConsol);

	LASInputFileName = fileName.toStdString();

	recentFileLocation = LASInputFileName.substr(0,LASInputFileName.find_last_of("/"));
}

void PetroLogTools::importDatabase()
{
	WorkingSetManager* manager = new WorkingSetManager(this);

	DataSelectorDialog* dialog = new DataSelectorDialog(this, manager);
	dialog->resize(550*2, 950);
	int code = dialog->exec();

	if (code==QDialog::Accepted) {
		PlotWellLog *pwl = new PlotWellLog(manager);
		pwl->show();
	}
}


void PetroLogTools::runPythonScript()
{
	workingDir = "/data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples";
	system("rm -rf /data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples/*");

	QString fileName = (workingDir + "/tmp.py").c_str();
  
	QFile file(fileName);
	file.open(QFile::WriteOnly | QFile::Text);

	QTextStream stream(&file);
	stream << mainConsol->toPlainText();

	file.flush();
	file.close();


	// Create launcher
	std::ofstream launcher (workingDir + "/launcher.bash");
  
	launcher << "source " + sourceFile << std::endl;
	launcher << "source /data/PLI/INTEGRATION/REDHAT7/xrai_v2_bnni_python/source.sh" << std::endl;
	launcher << "python "+ workingDir +"/tmp.py" << std::endl;
	
	launcher.close();

	// Launch python script
	mainProcess = new QProcess(this);
	mainProcess->setWorkingDirectory(workingDir.c_str());
	
	mainProcess->start(QString("bash"), QStringList((workingDir + "/launcher.bash").c_str()));
	
	connect(mainProcess,SIGNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
    connect(mainProcess,SIGNAL(readyReadStandardError()),this,SLOT(readyReadStandardError()));
}



void PetroLogTools::getLogNames()
{
	workingDir = "/data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples";
	system("rm -rf /data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples/*");

	// Create tmp python script
	std::ofstream pythonScript (workingDir + "/tmp.py");
  
	pythonScript << "import lasio" << std::endl;
	pythonScript << "log = lasio.read(\""+LASInputFileName+"\")" << std::endl;
	pythonScript << "print(log.keys())" << std::endl;
	pythonScript.close();


	// Create launcher
	std::ofstream launcher (workingDir + "/launcher.bash");
  
	launcher << "source " + sourceFile << std::endl;
	launcher << "source /data/PLI/INTEGRATION/REDHAT7/xrai_v2_bnni_python/source.sh" << std::endl;
	launcher << "python "+ workingDir +"/tmp.py" << std::endl;
	
	launcher.close();

	// Launch python script

    mainProcess = new QProcess(this);
	mainProcess->setWorkingDirectory(workingDir.c_str());
	
	mainProcess->start(QString("bash"), QStringList((workingDir + "/launcher.bash").c_str()));
	
	connect(mainProcess,SIGNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
    connect(mainProcess,SIGNAL(readyReadStandardError()),this,SLOT(readyReadStandardError()));
}

void PetroLogTools::clearOutputText()
{
	outputConsol->clear();
}

void PetroLogTools::clearInputText()
{
	mainConsol->clear();
}

void PetroLogTools::showLASFileName()
{
	outputConsol->setText(LASInputFileName.c_str());
}

void PetroLogTools::showTopsFileName()
{
	outputConsol->setText(formationTopsFileName.c_str());
}

void PetroLogTools::showLASFileContent()
{
	load_file_to_qtextedit(LASInputFileName.c_str(), outputConsol);
}

void PetroLogTools::showTopsFileContent()
{
	load_file_to_qtextedit(formationTopsFileName.c_str(), outputConsol);
}

void PetroLogTools::showWorkingDir()
{
	outputConsol->setText(workingDir.c_str());
}

void PetroLogTools::importFormationTops()
{
	QString fileName = QFileDialog::getOpenFileName(this,
    	tr("Open Formation Tops file"), recentFileLocation.c_str(), tr("Formation tops (*.csv)"));
	
	load_file_to_qtextedit(fileName, mainConsol);

	formationTopsFileName = fileName.toStdString();

	recentFileLocation = formationTopsFileName.substr(0,formationTopsFileName.find_last_of("/"));
}

void PetroLogTools::load_file_to_qtextedit(QString const fileName, QTextEdit *qte)
{
	QFile file(fileName);
	file.open(QFile::ReadOnly | QFile::Text);

	qte->setText(file.readAll());

	file.close();
}

void PetroLogTools::plotWellLogs()
{

	system( "source /data/PLI/NKDEEP/sytuan/Libs/DearImGui/source.sh");
	system( "cd /data/PLI/NKDEEP/sytuan/Libs/DearImGui/WellLogViewer/build && /data/PLI/NKDEEP/sytuan/Libs/DearImGui/WellLogViewer/build/WellLogViewer&");



	workingDir = "/data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples";
	system("rm -rf /data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples/*");

	// Create tmp python script for plotting
	std::ofstream pythonScript (workingDir + "/tmp.py");
  
	pythonScript << "import sys" << std::endl;
	pythonScript << "import petropy as ptr" << std::endl;
	pythonScript << "log = ptr.Log(\""+LASInputFileName+"\")" << std::endl;
	pythonScript << "viewer = ptr.LogViewer(log, top=6950, height=100)" << std::endl;
	pythonScript << "viewer.show()" << std::endl;
	pythonScript.close();

	// Create launcher
	std::ofstream launcher (workingDir + "/launcher.bash");
  
	launcher << "source " + sourceFile << std::endl;
	launcher << "source /data/PLI/INTEGRATION/REDHAT7/xrai_v2_bnni_python/source.sh" << std::endl;
	launcher << "python "+ workingDir +"/tmp.py" << std::endl;
	
	launcher.close();

	// Launch python script
	mainConsol->moveCursor (QTextCursor::End);

    mainProcess = new QProcess(this);
	mainProcess->setWorkingDirectory(workingDir.c_str());
	
	mainProcess->start(QString("bash"), QStringList((workingDir + "/launcher.bash").c_str()));
	
	connect(mainProcess,SIGNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
    connect(mainProcess,SIGNAL(readyReadStandardError()),this,SLOT(readyReadStandardError()));
}

void PetroLogTools::computeFormationFluidProperties()
{
	workingDir = "/data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples";
	system("rm -rf /data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples/*");

	// Create tmp python script for plotting
	std::ofstream pythonScript (workingDir + "/tmp.py");
  
	pythonScript << "import sys" << std::endl;
	pythonScript << "import petropy as ptr" << std::endl;
	pythonScript << "log = ptr.Log(\"" + LASInputFileName + "\")" << std::endl;
	pythonScript << "print(log.curves)" << std::endl;
	pythonScript << "log.tops_from_csv(\"" + formationTopsFileName + "\")" << std::endl;
	pythonScript << "print(log.tops)" << std::endl;
	pythonScript << "log.fluid_properties_parameters_from_csv()" << std::endl;
	pythonScript << "print(log.fluid_properties_parameters.keys())" << std::endl;
	pythonScript << "print(log.fluid_properties_parameters)" << std::endl;
	pythonScript << "log.formation_fluid_properties(log.tops.keys(), parameter = \'default\')" << std::endl;
	pythonScript << "print(log.curves)" << std::endl;
	pythonScript << "log.write(\"" + workingDir + "/tmp.las\")" << std::endl;
	pythonScript.close();


	// Create launcher
	std::ofstream launcher (workingDir + "/launcher.bash");
  
	launcher << "source " + sourceFile << std::endl;
	launcher << "source /data/PLI/INTEGRATION/REDHAT7/xrai_v2_bnni_python/source.sh" << std::endl;
	launcher << "python "+ workingDir +"/tmp.py" << std::endl;
	
	launcher.close();

	// Launch python script
	//mainConsol->moveCursor (QTextCursor::End);

    mainProcess = new QProcess(this);
	mainProcess->setWorkingDirectory(workingDir.c_str());
	
	mainProcess->start(QString("bash"), QStringList((workingDir + "/launcher.bash").c_str()));
	
	connect(mainProcess,SIGNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
    connect(mainProcess,SIGNAL(readyReadStandardError()),this,SLOT(readyReadStandardError()));
}

void PetroLogTools::computeFormationMultiminerals()
{
	workingDir = "/data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples";
	system("rm -rf /data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples/*");

	// Create tmp python script for plotting
	std::ofstream pythonScript (workingDir + "/tmp.py");
  
	pythonScript << "import sys" << std::endl;
	pythonScript << "import petropy as ptr" << std::endl;
	pythonScript << "log = ptr.Log(\""+LASInputFileName+"\")" << std::endl;
	pythonScript << "print(log.curves)" << std::endl;
	pythonScript << "log.tops_from_csv(\"" + formationTopsFileName + "\")" << std::endl;
	pythonScript << "print(log.tops)" << std::endl;
	pythonScript << "log.multimineral_parameters_from_csv()" << std::endl;
	pythonScript << "print(log.multimineral_parameters.keys())" << std::endl;
	pythonScript << "print(log.multimineral_parameters)" << std::endl;
	//pythonScript << "log.formation_multimineral_model(log.tops.keys(), parameter = \'default\')" << std::endl;
	pythonScript << "print(log.curves)" << std::endl;
	pythonScript << "log.write(\"" + workingDir + "/tmp.las\")" << std::endl;

	pythonScript.close();


	// Create launcher
	std::ofstream launcher (workingDir + "/launcher.bash");
  
	launcher << "source " + sourceFile << std::endl;
	launcher << "source /data/PLI/INTEGRATION/REDHAT7/xrai_v2_bnni_python/source.sh" << std::endl;
	launcher << "python "+ workingDir +"/tmp.py" << std::endl;
	
	launcher.close();

	// Launch Python script
	//mainConsol->moveCursor (QTextCursor::End);

    mainProcess = new QProcess(this);
	mainProcess->setWorkingDirectory(workingDir.c_str());
	
	mainProcess->start(QString("bash"), QStringList((workingDir + "/launcher.bash").c_str()));
	
	connect(mainProcess,SIGNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
    connect(mainProcess,SIGNAL(readyReadStandardError()),this,SLOT(readyReadStandardError()));
}

void PetroLogTools::readyReadStandardOutput(){
	outputConsol->setText(mainProcess->readAllStandardOutput());
	outputConsol->moveCursor (QTextCursor::End);
}

void PetroLogTools::readyReadStandardError(){
    outputConsol->setText(mainProcess->readAllStandardError());
	outputConsol->moveCursor (QTextCursor::End);
}

void PetroLogTools::createActions()
{
	importLASAct = new QAction(tr("Import a LAS input"), this);
    importLASAct->setStatusTip(tr("Import a LAS file"));
    connect(importLASAct, &QAction::triggered, this, &PetroLogTools::importLAS);

	importDatabaseAct = new QAction(tr("Import from database"), this);
    importDatabaseAct->setStatusTip(tr("Import data from database"));
    connect(importDatabaseAct, &QAction::triggered, this, &PetroLogTools::importDatabase);

	getLogNamesAct = new QAction(tr("Log Names"), this);
    getLogNamesAct->setStatusTip(tr("Show log names"));
    connect(getLogNamesAct, &QAction::triggered, this, &PetroLogTools::getLogNames);

	importFormationTopsAct = new QAction(tr("Import formation tops"), this);
    importFormationTopsAct->setStatusTip(tr("Import a csv file for forpation tops"));
    connect(importFormationTopsAct, &QAction::triggered, this, &PetroLogTools::importFormationTops);

    clearInputTextAct = new QAction(tr("Clear input"), this);
    clearInputTextAct->setStatusTip(tr("Clear the input text"));
    connect(clearInputTextAct, &QAction::triggered, this, &PetroLogTools::clearInputText);

	clearOutputTextAct = new QAction(tr("Clear output"), this);
    clearOutputTextAct->setStatusTip(tr("Clear the output text"));
    connect(clearOutputTextAct, &QAction::triggered, this, &PetroLogTools::clearOutputText);

	showLASFileNameAct = new QAction(tr("Show LAS file name"), this);
    showLASFileNameAct->setStatusTip(tr("Show the name of the actual LAS file"));
    connect(showLASFileNameAct, &QAction::triggered, this, &PetroLogTools::showLASFileName);

	showLASFileContentAct = new QAction(tr("Show LAS file content"), this);
    showLASFileContentAct->setStatusTip(tr("Show the content of the actual LAS file"));
    connect(showLASFileContentAct, &QAction::triggered, this, &PetroLogTools::showLASFileContent);

	showTopsFileNameAct = new QAction(tr("Show tops file name"), this);
    showTopsFileNameAct->setStatusTip(tr("Show the name of the tops file"));
    connect(showTopsFileNameAct, &QAction::triggered, this, &PetroLogTools::showTopsFileName);

	showTopsFileContentAct = new QAction(tr("Show tops file content"), this);
    showTopsFileContentAct->setStatusTip(tr("Show the content of the actual tops file"));
    connect(showTopsFileContentAct, &QAction::triggered, this, &PetroLogTools::showTopsFileContent);

	showWorkingDirAct = new QAction(tr("Working directory"), this);
    showWorkingDirAct->setStatusTip(tr("Show the working directory"));
    connect(showWorkingDirAct, &QAction::triggered, this, &PetroLogTools::showWorkingDir);

    plotWellLogsAct = new QAction(tr("Plot"), this);
    plotWellLogsAct->setStatusTip(tr("Plot well logs"));
    connect(plotWellLogsAct, &QAction::triggered, this, &PetroLogTools::plotWellLogs);

	runPythonScriptAct = new QAction(tr("Run python script"), this);
	runPythonScriptAct->setShortcuts(QKeySequence::Refresh);
	runPythonScriptAct->setStatusTip(tr("Run the actual python script"));
	connect(runPythonScriptAct, &QAction::triggered, this, &PetroLogTools::runPythonScript);

	computeFormationFluidPropertiesAct = new QAction(tr("Formation Fluid Properties"), this);
    computeFormationFluidPropertiesAct->setStatusTip(tr("Compute formation fluid properties"));
    connect(computeFormationFluidPropertiesAct, &QAction::triggered, this, &PetroLogTools::computeFormationFluidProperties);

	computeFormationMultimineralsAct = new QAction(tr("Formation Multiminerals"), this);
    computeFormationMultimineralsAct->setStatusTip(tr("Compute formation multiminerals"));
    connect(computeFormationMultimineralsAct, &QAction::triggered, this, &PetroLogTools::computeFormationMultiminerals);

}

void PetroLogTools::createMenus()
{
	projectMenu = menuBar()->addMenu(tr("Project"));
	projectMenu->addAction(importLASAct);
	projectMenu->addAction(importFormationTopsAct);
	projectMenu->addAction(importDatabaseAct);

    logMenu = menuBar()->addMenu(tr("Well Logs"));
    logMenu->addAction(getLogNamesAct);

    computeMenu = menuBar()->addMenu(tr("Compute"));
	computeMenu->addAction(runPythonScriptAct);
    computeMenu->addAction(computeFormationFluidPropertiesAct);
	computeMenu->addAction(computeFormationMultimineralsAct);

	visualizationMenu = menuBar()->addMenu(tr("Visualization"));
	visualizationMenu->addAction(clearInputTextAct);
	visualizationMenu->addAction(clearOutputTextAct);
    visualizationMenu->addAction(plotWellLogsAct);
	visualizationMenu->addAction(showLASFileNameAct);
	visualizationMenu->addAction(showLASFileContentAct);
	visualizationMenu->addAction(showTopsFileNameAct);
	visualizationMenu->addAction(showTopsFileContentAct);
	visualizationMenu->addAction(showWorkingDirAct);
}


