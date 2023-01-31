/*
 *
 *
 *  Created on: 30 May 2022
 *      Author: l0359127
 */

#include "GeomechanicsTools.h"

GeomechanicsTools::GeomechanicsTools()
{
    QWidget *widget = new QWidget;
    setCentralWidget(widget);

    QWidget *topFiller = new QWidget;
    topFiller->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    mainConsol = new QTextEdit();
    mainConsol->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
    //mainConsol->setAlignment(Qt::AlignCenter);

    QWidget *bottomFiller = new QWidget;
    bottomFiller->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    layout = new QVBoxLayout;
    layout->setContentsMargins(5, 5, 5, 5);
    //layout->addWidget(topFiller);
    layout->addWidget(mainConsol);
    //layout->addWidget(bottomFiller);
    widget->setLayout(layout);

    createActions();
    createMenus();

    QString message = tr("Tools for geomechanical simulation");
    statusBar()->showMessage(message);

    setWindowTitle(tr("Geomechanics"));
    setMinimumSize(160, 160);
    resize(1012, 683);
}

#ifndef QT_NO_CONTEXTMENU
void GeomechanicsTools::contextMenuEvent(QContextMenuEvent *event)
{
    QMenu menu(this);
	menu.addAction(openProjectAct);
    menu.addAction(runSimulationAct);
	menu.addAction(saveAndRunSimulationAct);
	menu.addAction(visualization3DAct);
    menu.exec(event->globalPos());
}
#endif // QT_NO_CONTEXTMENU

void GeomechanicsTools::openProject()
{
	QString fileName = QFileDialog::getOpenFileName(this,
    	tr("Open XML input file"), recentFileLocation.c_str(), tr("Input XML Files (*.xml)"));
	
	load_file_to_qtextedit(fileName, mainConsol);

	xmlInputFileName = fileName.toStdString();

	workingDir = xmlInputFileName.substr(0,xmlInputFileName.find_last_of("/"));

	if(workingDir.find("/data/PLI/NKDEEP/sytuan/Libs/Geosx/GEOSX/inputFiles") != std::string::npos)
	{
		workingDir = "/data/PLI/NKDEEP/sytuan/Libs/Geosx/caseStudies/rerunIntegratedTests";
	}

	recentFileLocation = xmlInputFileName.substr(0,xmlInputFileName.find_last_of("/"));
}

void GeomechanicsTools::openBaseXML()
{
	QString fileName = QFileDialog::getOpenFileName(this,
    	tr("Open base XML file"), recentFileLocation.c_str(), tr("Base XML file (*_base.xml)"));
	
	load_file_to_qtextedit(fileName, mainConsol);

	workingDir = xmlInputFileName.substr(0,xmlInputFileName.find_last_of("/"));

	if(workingDir.find("/data/PLI/NKDEEP/sytuan/Libs/Geosx/GEOSX/inputFiles") != std::string::npos)
	{
		workingDir = "/data/PLI/NKDEEP/sytuan/Libs/Geosx/caseStudies/rerunIntegratedTests";
	}

	recentFileLocation = xmlInputFileName.substr(0,xmlInputFileName.find_last_of("/"));
}

void GeomechanicsTools::load_file_to_qtextedit(QString const fileName, QTextEdit *qte)
{
	QFile file(fileName);
	file.open(QFile::ReadOnly | QFile::Text);

	qte->setText(file.readAll());
	file.close();
}

void GeomechanicsTools::generateMesh()
{
	MeshGenerator *p = new MeshGenerator();
	p->setVisible(true);
}

void GeomechanicsTools::readGRDECL()
{
	ReadGRDECL *p = new ReadGRDECL();
	p->setVisible(true);
}

void GeomechanicsTools::runSimulation()
{
	// Clean the workingDir if rerun the integrated tests
	if(workingDir == "/data/PLI/NKDEEP/sytuan/Libs/Geosx/caseStudies/rerunIntegratedTests/")
	{
		system("rm -rf /data/PLI/NKDEEP/sytuan/Libs/Geosx/caseStudies/rerunIntegratedTests/*");
	}

	// Create launcher
	std::ofstream launcher (workingDir + "/launcher.bash");
  
	launcher << "source " + sourceFile << std::endl;
	launcher << geosxPath + " -i " + xmlInputFileName << std::endl;
	
	launcher << "ls " + workingDir;

	launcher.close();

	// Run the simulation
    mainProcess = new QProcess(this);
	mainProcess->setWorkingDirectory(workingDir.c_str());
	
	mainProcess->start(QString("bash"), QStringList((workingDir + "/launcher.bash").c_str()));
	
	connect(mainProcess,SIGNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
    connect(mainProcess,SIGNAL(readyReadStandardError()),this,SLOT(readyReadStandardError()));
}

void GeomechanicsTools::saveAndRunSimulation()
{
	// Create tmp input xml
	std::string xmlInputFileNameDir = xmlInputFileName.substr(0,xmlInputFileName.find_last_of("/"));

	xmlInputFileName = xmlInputFileNameDir + "/tmp.xml";
	QString fileName = xmlInputFileName.c_str();
  
	QFile file(fileName);
	file.open(QFile::WriteOnly | QFile::Text);

	QTextStream stream(&file);
	stream << mainConsol->toPlainText();

	file.flush();
	file.close();

	runSimulation();
}

void GeomechanicsTools::readyReadStandardOutput(){
	mainConsol->insertPlainText(mainProcess->readAllStandardOutput());
	mainConsol->moveCursor (QTextCursor::End);
}

void GeomechanicsTools::readyReadStandardError(){
    mainConsol->insertPlainText(mainProcess->readAllStandardError());
	mainConsol->moveCursor (QTextCursor::End);
}

void GeomechanicsTools::terzaghiExample()
{
	workingDir = "/data/PLI/NKDEEP/sytuan/Libs/Geosx/caseStudies/rerunIntegratedTests/";
    xmlInputFileName = "/data/PLI/NKDEEP/sytuan/Libs/Geosx/GEOSX/inputFiles/poromechanics/PoroElastic_Terzaghi_smoke.xml";

	showXMLInput();
}

void GeomechanicsTools::rhobExample()
{
	workingDir = "/data/PLI/NKDEEP/sytuan/caseStudies/";
	xmlInputFileName = "/data/PLI/NKDEEP/sytuan/caseStudies/rhob_geosx.xml";

	showXMLInput();
}

void GeomechanicsTools::showXMLInput()
{
	QFile file(xmlInputFileName.c_str());
	file.open(QFile::ReadOnly | QFile::Text);

	mainConsol->setText(file.readAll());
	file.close();
}

void GeomechanicsTools::visualization3D()
{
	system("/data/appli_PITSI/MAJIX2018/PROD/ParaviewLauncherGUI/paraviewLauncherGUI.sh");
}

void GeomechanicsTools::createActions()
{
	openProjectAct = new QAction(tr("&Open a project"), this);
    //openProjectAct->setShortcuts(QKeySequence::Open);
    openProjectAct->setStatusTip(tr("Open a geomechanical project"));
    connect(openProjectAct, &QAction::triggered, this, &GeomechanicsTools::openProject);

	openBaseXMLAct = new QAction(tr("Open a base XML file"), this);
    openBaseXMLAct->setStatusTip(tr("Open a base XML file"));
    connect(openBaseXMLAct, &QAction::triggered, this, &GeomechanicsTools::openBaseXML);

    generateMeshAct = new QAction(tr("&Generate Mesh"), this);
    //generateMeshAct->setShortcuts(QKeySequence::New);
    generateMeshAct->setStatusTip(tr("Generate a mesh"));
    connect(generateMeshAct, &QAction::triggered, this, &GeomechanicsTools::generateMesh);

    readGRDECLAct = new QAction(tr("&Read GRDECL Mesh"), this);
    readGRDECLAct->setShortcuts(QKeySequence::Open);
    readGRDECLAct->setStatusTip(tr("Read GRDECL mesh format"));
    connect(readGRDECLAct, &QAction::triggered, this, &GeomechanicsTools::readGRDECL);

    runSimulationAct = new QAction(tr("&Run"), this);
    runSimulationAct->setStatusTip(tr("Run the simulation"));
    connect(runSimulationAct, &QAction::triggered, this, &GeomechanicsTools::runSimulation);

	saveAndRunSimulationAct = new QAction(tr("&Save & Run"), this);
    saveAndRunSimulationAct->setStatusTip(tr("Save modified XML and Run the simulation"));
    connect(saveAndRunSimulationAct, &QAction::triggered, this, &GeomechanicsTools::saveAndRunSimulation);

	visualization3DAct = new QAction(tr("&Paraview"), this);
    visualization3DAct->setStatusTip(tr("3D Visualization"));
    connect(visualization3DAct, &QAction::triggered, this, &GeomechanicsTools::visualization3D);

    terzaghiExampleAct = new QAction(tr("&Terzaghi Problem"), this);
    terzaghiExampleAct->setCheckable(true);
    terzaghiExampleAct->setStatusTip(tr("Example of Terzaghi poroelastic problem"));
    connect(terzaghiExampleAct, &QAction::triggered, this, &GeomechanicsTools::terzaghiExample);

	rhobExampleAct = new QAction(tr("&Rhob Problem"), this);
    rhobExampleAct->setCheckable(true);
    rhobExampleAct->setStatusTip(tr("Rhob example"));
    connect(rhobExampleAct, &QAction::triggered, this, &GeomechanicsTools::rhobExample);
}

void GeomechanicsTools::createMenus()
{
	projectMenu = menuBar()->addMenu(tr("Project"));
	projectMenu->addAction(openProjectAct);
	projectMenu->addAction(openBaseXMLAct);

    meshMenu = menuBar()->addMenu(tr("&Mesh"));
    meshMenu->addAction(generateMeshAct);
    meshMenu->addAction(readGRDECLAct);

    simulationMenu = menuBar()->addMenu(tr("&Simulation"));
    simulationMenu->addAction(runSimulationAct);
	simulationMenu->addAction(saveAndRunSimulationAct);//run modified xml

	examplesMenu = menuBar()->addMenu(tr("&Examples"));
	poroelasticExamplesMenu = examplesMenu->addMenu(tr("&PoroElastic"));
    poroelasticExamplesMenu->addAction(terzaghiExampleAct);
	poroelasticExamplesMenu->addAction(rhobExampleAct);


	visualizationMenu = menuBar()->addMenu(tr("&Visualization"));
    visualizationMenu->addAction(visualization3DAct);
}
