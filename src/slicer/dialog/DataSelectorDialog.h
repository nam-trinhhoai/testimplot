#ifndef DataSelectorDialog_H
#define DataSelectorDialog_H

#include <QDialog>
#include <QMutex>

#include "seismic3dabstractdataset.h"
#include <WellUtil.h>
#include "utils/stringutil.h"
#include <boost/filesystem.hpp>

class QComboBox;
class SeismicSurvey;
class Seismic3DAbstractDataset;
class GeotimeProjectManagerWidget;
class ProjectManagerWidget;
class WorkingSetManager;
class WellBore;
class WellHead;

class DataSelectorDialog: public QDialog {
Q_OBJECT
public:
	DataSelectorDialog(QWidget *parent, WorkingSetManager *manager, int flag = 0);

	virtual ~DataSelectorDialog();


	// There is no naming control for both function
	static QString getSismageNameFromSeismicFile(QString seismicFile);
	static QString getSismageNameFromDescFile(QString descFile);

	// code to transfert tarum session to next vison session
	// ! warning does not worry about overwriting files !
//	void run();
	GeotimeProjectManagerWidget* getSelector(){ return m_selectorWidget;}
	static SeismicSurvey* dataGetBaseSurvey(WorkingSetManager *pWorkingSetManager,QString surveyName,QString surveyPath,bool &rbIsNewSurvey);
	static void createSeismic(SeismicSurvey* baseSurvey,WorkingSetManager *pWorkingSetManager,const std::vector<QString>& datasetPaths ,
			const std::vector<QString>& datasetNames,bool &rbIsNewSurvey,bool bUpdateDataSet=true);
	// addHorizon, vector copy is expected to allow the use of filterHorizons without doing the copy in the function
	static void addHorizon(WorkingSetManager *pWorkingSetManager,std::vector<QString> horizonPaths ,std::vector<QString> horizonNames,QString seismicDirPath);
	static void addWellBore(WorkingSetManager *pWorkingSetManager,const std::vector<WELLLIST>& wells,
			const std::vector<MARKER>& wellPicksList, bool bUpdateWellBore=true);
	static void createPicks(WorkingSetManager *pWorkingSetManager,const std::vector<MARKER>& wellPicks,
			QString wellHeadPath, QString wellBorePath, WellBore* bore);

	static void addNurbs(WorkingSetManager *pWorkingSetManager,std::vector<QString> nurbspath,std::vector<QString> nurbsNames);
	/**
	 * to speed up welllist can be modfied as :
	 * std::vector<Marker> welllist
	 * Marker :
	 *  - std::vector<WELLPICKSLIST>
	 *  - marker name
	 */
	static void deleteUnselectedPick(WellBore* bore,const std::vector<MARKER>& welllist,
			QString wellHeadPath,QString wellBorePath);
	static WellBore *createUpdatebore(const WELLBORELIST& wellbore,long index,WorkingSetManager *pWorkingSetManager);
	static void removeUnselectedWells(WorkingSetManager *pWorkingSetManager,const std::vector<WELLLIST>& wells,bool bUpdateWellBore=true);
	static bool addWellHead(const WELLLIST& well,WorkingSetManager *pWorkingSetManager,long &rIndex);



	static QFileInfoList getFiles(std::string path, QStringList ext);


	static bool addNVHorizons(WorkingSetManager *pWorkingSetManager, SeismicSurvey *survey, std::vector<QString> horizonPaths, std::vector<QString> horizonName);
	// static bool removeNVHorizons(WorkingSetManager *pWorkingSetManager, SeismicSurvey *survey, std::vector<QString> horizonPaths, std::vector<QString> horizonName);
	static bool removeNVHorizonsAttribut(WorkingSetManager *pWorkingSetManager, SeismicSurvey *survey,
			QString horizonPath, QString horizonName,
			QString attributPath, QString attributName);


	static bool addNVIsoHorizons(WorkingSetManager *pWorkingSetManager, SeismicSurvey *survey, std::vector<QString> horizonPaths, std::vector<QString> horizonName);
	static bool addNVHorizonAnims(WorkingSetManager *pWorkingSetManager, SeismicSurvey *survey, std::vector<QString> animPaths, std::vector<QString> animNames);
	GeotimeProjectManagerWidget *getSelectorWidget();
	ProjectManagerWidget *getSelectorWidget2() { return m_selectorWidget2; }


public slots:
	void saveSession();

protected:
	QMutex m_acceptMutex;
	GeotimeProjectManagerWidget* m_selectorWidget = nullptr;
	ProjectManagerWidget *m_selectorWidget2 = nullptr;
	WorkingSetManager *m_manager;
	static::Seismic3DAbstractDataset* appendDataset(
			WorkingSetManager *pManager,
			SeismicSurvey *baseSurvey,
			const QString &datasetPath, const QString &datasetName,
			Seismic3DAbstractDataset::CUBE_TYPE type,
			bool forceCPU);
	static void removeDataset(SeismicSurvey *baseSurvey,const QString &datasetName);
	static void filterHorizons(std::vector<QString>& horizonNames,
			std::vector<QString>& horizonPaths,
			std::vector<QString>& horizonExtractionDataPaths,QString seismicDirPath);
private slots:
	void accepted();
	void loadSession();
};

#endif
