/*
 *
 *
 *  Created on: 07 June 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PETROLOGTOOLS_H_
#define NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PETROLOGTOOLS_H_

#include "ReadGRDECL.h"
#include "MeshGenerator.h"
#include "PlotWellLog.h"

#include "workingsetmanager.h"
#include "DataSelectorDialog.h"
#include "geotimegraphicsview.h"
#include "folderdata.h"
#include "wellhead.h"
#include "wellbore.h"


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


class PetroLogTools : public QMainWindow
{
    Q_OBJECT

public:
    PetroLogTools();

protected:
#ifndef QT_NO_CONTEXTMENU
    void contextMenuEvent(QContextMenuEvent *event) override;
#endif // QT_NO_CONTEXTMENU

private slots:
	void readyReadStandardOutput();

	void readyReadStandardError();

	void importLAS();
	void importDatabase();
	void importFormationTops();
    void plotWellLogs();
	void computeFormationFluidProperties();
	void computeFormationMultiminerals();
	
	void getLogNames();
	void runPythonScript();

	void clearOutputText();
	void clearInputText();
	void showLASFileName();
	void showLASFileContent();
	void showTopsFileName();
	void showTopsFileContent();
	void showWorkingDir();
private:

void geotimeLaunch();

	void load_file_to_qtextedit(QString const fileName, QTextEdit *qte);
    void createActions();
    void createMenus();
	void showXMLInput();

	QProcess *mainProcess;
	QHBoxLayout *layout;
	QMenu *projectMenu;
    QMenu *logMenu;
    QMenu *computeMenu;
 	QMenu *visualizationMenu;
	
    QAction *plotWellLogsAct;
	QAction *computeFormationFluidPropertiesAct;
	QAction *computeFormationMultimineralsAct;
	QAction *importLASAct;
	QAction *importDatabaseAct;
	QAction *importFormationTopsAct;	
	QAction *getLogNamesAct;
	QAction *runPythonScriptAct;
	QAction *clearOutputTextAct;
	QAction *clearInputTextAct;
	QAction *showLASFileNameAct;
	QAction *showLASFileContentAct;
	QAction *showTopsFileNameAct;
	QAction *showTopsFileContentAct;
	QAction *showWorkingDirAct;

    QTextEdit *mainConsol;
	QTextEdit *outputConsol;

	std::string workingDir = "/data/PLI/NKDEEP/sytuan/Libs/PetroPy/caseStudies/runExamples";
	std::string LASInputFileName;
	std::string formationTopsFileName;	

	std::string const geosxPath = "/data/PLI/NKDEEP/sytuan/Libs/Geosx/GEOSX/build-xraiTotal-release/bin/geosx";
	std::string const sourceFile = "/data/PLI/NKDEEP/sytuan/buildGuide/source.sh";
	std::string recentFileLocation = "/data/PLI/NKDEEP/sytuan/Libs/PetroPy/PetroPy/examples";
};

#endif // NEXTVISION_SRC_WIDGET_WGEOMECHANICS_PETROLOGTOOLS_H_
