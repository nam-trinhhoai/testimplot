/*
 *
 *
 *  Created on: 30 May 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_GEOMECHANICSTOOLS_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_GEOMECHANICSTOOLS_H_

#include "ReadGRDECL.h"
#include "MeshGenerator.h"

#include <QMainWindow>
#include <QProcess>
#include <QTextCursor>
#include <QFile>
#include <QtWidgets>


QT_BEGIN_NAMESPACE
class QAction;
class QActionGroup;
class QLabel;
class QMenu;
QT_END_NAMESPACE

class GeomechanicsTools : public QMainWindow
{
    Q_OBJECT

public:
    GeomechanicsTools();

protected:
#ifndef QT_NO_CONTEXTMENU
    void contextMenuEvent(QContextMenuEvent *event) override;
#endif // QT_NO_CONTEXTMENU

private slots:
	void readyReadStandardOutput();

	void readyReadStandardError();

	void openProject();
	void openBaseXML();
    void generateMesh();
    void readGRDECL();
    void runSimulation();
	void saveAndRunSimulation();
    void terzaghiExample();
	void rhobExample();
	void visualization3D();

private:

	void load_file_to_qtextedit(QString const fileName, QTextEdit *qte);
    void createActions();
    void createMenus();
	void showXMLInput();
	QProcess *mainProcess;
	QVBoxLayout *layout;
	QMenu *projectMenu;
    QMenu *meshMenu;
    QMenu *simulationMenu;
    QMenu *examplesMenu;
	QMenu *poroelasticExamplesMenu;
	QMenu *visualizationMenu;
    QAction *generateMeshAct;
    QAction *readGRDECLAct;
    QAction *runSimulationAct;
	QAction *saveAndRunSimulationAct;
	QAction *openProjectAct;
	QAction *openBaseXMLAct;
    QAction *terzaghiExampleAct;
	QAction *rhobExampleAct;
	QAction *visualization3DAct;
    QTextEdit *mainConsol;

	std::string workingDir;
	std::string xmlInputFileName;	

	std::string const geosxPath = "/data/PLI/NKDEEP/sytuan/Libs/Geosx/GEOSX/build-xraiTotal-release/bin/geosx";
	std::string const sourceFile = "/data/PLI/NKDEEP/sytuan/Libs/Geosx/source.sh";
	std::string recentFileLocation = "/data/PLI/NKDEEP/sytuan/Libs/Geosx/GEOSX/inputFiles/";
};

#endif // NEXTVISION_SRC_WIDGET_WGEOMECHANICS_GEOMECHANICSTOOLS_H_
