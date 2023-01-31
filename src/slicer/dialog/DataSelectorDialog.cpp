#include "DataSelectorDialog.h"

#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <QGridLayout>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QLabel>
#include <QSettings>
#include <QStyledItemDelegate>
#include <QFileDialog>
#include <QProcess>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <chrono>

#include <QRegularExpression>
#include "Xt.h"

#include "DataSelectorDialog.h"
#include "seismicsurvey.h"
#include "smsurvey3D.h"
#include "seismic3ddataset.h"
#include "seismic3dcudadataset.h"
#include "smdataset3D.h"
#include "GeotimeProjectManagerWidget.h"
#include "ProjectManagerWidget.h"
#include "seismic3dabstractdataset.h"
#include "workingsetmanager.h"
#include "ijkhorizon.h"
#include "wellhead.h"
#include "wellbore.h"
#include "wellpick.h"
#include "folderdata.h"
#include "globalconfig.h"
#include "horizondatarep.h"
#include "horizonfolderdata.h"
#include "marker.h"
#include <freehorizon.h>
#include <freeHorizonQManager.h>
#include <isohorizon.h>

using namespace std::chrono;

const QLatin1String RGT_SEISMIC_SLICER_DIR_PROJECT("RGTSeismicSlicer/DirProject");
const QLatin1String RGT_SEISMIC_SLICER_PROJECT("RGTSeismicSlicer/Project");
const QLatin1String RGT_SEISMIC_SLICER_SURVEY_PATH("RGTSeismicSlicer/SurveyPath");

DataSelectorDialog::DataSelectorDialog(	QWidget *parent, WorkingSetManager *manager, int flag) :
								QDialog(parent), m_manager( manager){
	QString title="Data Selection";

	setWindowTitle(title);

	QVBoxLayout * mainLayout=new QVBoxLayout(this);

	QPushButton* loadSessionButton = new QPushButton("Load session");
	mainLayout->addWidget(loadSessionButton);

	if ( flag == 0 )
	{
		m_selectorWidget = new GeotimeProjectManagerWidget(this);
		mainLayout->addWidget(m_selectorWidget);
	}
	else
	{
		m_selectorWidget2 = new ProjectManagerWidget(false);
		mainLayout->addWidget(m_selectorWidget2);
	}
	/*
	m_selectorWidget->setVisible(false);
	ProjectManagerWidget *m_selectorWidget2 = new ProjectManagerWidget();
	mainLayout->addWidget(m_selectorWidget2);
	*/

	QHBoxLayout* sessionLayout = new QHBoxLayout;
	QPushButton* saveSessionButton = new QPushButton("Save session");
	sessionLayout->addWidget(saveSessionButton);

	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(buttonBox, SIGNAL(accepted()), this, SLOT(accepted()));
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
	sessionLayout->addWidget(buttonBox);
	mainLayout->addLayout(sessionLayout);

	//    QSettings settings;
	//    const QString dirProject = settings.value(RGT_SEISMIC_SLICER_DIR_PROJECT,
	//                                            "").toString();
	//    m_selectorWidget.
	//    const QString project = settings.value(RGT_SEISMIC_SLICER_PROJECT,
	//                                            "").toString();
	//    const QString lastPath = settings.value(RGT_SEISMIC_SLICER_SURVEY_PATH,
	//                                            QDir::homePath()).toString();
	//	QString path(lastPath);
	connect(loadSessionButton, &QPushButton::clicked, this, &DataSelectorDialog::loadSession);
	connect(saveSessionButton, &QPushButton::clicked, this, &DataSelectorDialog::saveSession);

	// code to transfert tarum session to next vison session
	// ! warning does not worry about overwriting files !
	//	run();
}

DataSelectorDialog::~DataSelectorDialog() {

}

GeotimeProjectManagerWidget *DataSelectorDialog::getSelectorWidget()
{
	return m_selectorWidget;
}

QString DataSelectorDialog::getSismageNameFromSeismicFile(QString seismicFile) {
	QFileInfo info(seismicFile);
	QString fullName = info.fileName();
	QString path = info.absolutePath();

	int lastPoint = fullName.lastIndexOf(".");
	QString fileNameNoExt = fullName.left(lastPoint);
	QString ext = fullName.right(fullName.size()-lastPoint-1);
	QString desc_filename = path + "/" + fileNameNoExt + ".desc";
	QString sismageName = getSismageNameFromDescFile(desc_filename);

	if ( sismageName.isNull() || sismageName.isEmpty() ) {
		QString tmp = fullName;

		tmp = tmp.left(lastPoint);
		QString header = tmp.left(10);
		if ( header.compare(QString("seismic3d.")) == 0 )
		{
			tmp.remove(0, 10);
		}

		sismageName = tmp;
	}
	return sismageName;
}

QString DataSelectorDialog::getSismageNameFromDescFile(QString descFile) {
	QString sismageName;

	char buff[1000];
	FILE *pfile = fopen(descFile.toStdString().c_str(), "r");

	if ( pfile != NULL ) {
		fgets(buff, 10000, pfile);
		fgets(buff, 10000, pfile);
		fgets(buff, 10000, pfile);
		buff[0] = 0; fscanf(pfile, "name=%s\n", buff);
		fclose(pfile);
		QString tmp = QString(buff);
		if ( !tmp.isEmpty() )
		{
			sismageName = QString(tmp);
		}
	}
	return sismageName;
}

Seismic3DAbstractDataset* DataSelectorDialog::appendDataset(
		WorkingSetManager *pManager,
		SeismicSurvey *baseSurvey,
		const QString &datasetPath, const QString &datasetName,
		Seismic3DAbstractDataset::CUBE_TYPE type,
		bool forceCPU) {
	int width, depth, heigth;
	{
		inri::Xt xt(datasetPath.toStdString());
		if (!xt.is_valid()) {
			//todo std::cerr << "xt cube is not valid (" << path << ")" << std::endl;
			return nullptr;
		}

		width = xt.nRecords();
		depth = xt.nSlices();
		heigth = xt.nSamples();
	}

	size_t avail;
	size_t total;
	cudaMemGetInfo(&avail, &total);

	size_t cubeSize = width * heigth * depth * sizeof(short);
	Seismic3DAbstractDataset *seismic;
	Seismic3DDataset::CUBE_TYPE cubeType = (datasetName.contains(QRegularExpression("_[rR][gG][tT]"))) ? Seismic3DDataset::CUBE_TYPE::RGT:Seismic3DDataset::CUBE_TYPE::Seismic;
	if (cubeType==Seismic3DDataset::CUBE_TYPE::Seismic) {
		cubeType = (datasetName.contains(QRegularExpression("[nN][eE][xX][tT][vV][iI][sS][iI][oO][nN][pP][aA][tT][cC][hH]"))) ? Seismic3DDataset::CUBE_TYPE::Patch:Seismic3DDataset::CUBE_TYPE::Seismic;
	}
	if (cubeSize < avail - avail * 5 / 100 && !forceCPU) {
		seismic = new Seismic3DCUDADataset(baseSurvey, datasetName,pManager, cubeType);
	} else {
		std::cout << "CPU Loading" << std::endl;
		seismic = new Seismic3DDataset(baseSurvey, datasetName, pManager,cubeType, datasetPath);
	}
	seismic->loadFromXt(datasetPath.toStdString());

	SmDataset3D d3d(datasetPath.toStdString());
	seismic->setIJToInlineXlineTransfo(d3d.inlineXlineTransfo());
	seismic->setIJToInlineXlineTransfoForInline(
			d3d.inlineXlineTransfoForInline());
	seismic->setIJToInlineXlineTransfoForXline(
			d3d.inlineXlineTransfoForXline());
	seismic->setSampleTransformation(d3d.sampleTransfo());

	baseSurvey->addDataset(seismic);
	return seismic;
}

void DataSelectorDialog::removeDataset(SeismicSurvey *baseSurvey,const QString &datasetName) {

	QList<Seismic3DAbstractDataset*> dataSetList = baseSurvey->datasets();
	for (int i = 0; i < dataSetList.size(); ++i) {
		if (dataSetList.at(i)->name() == datasetName){
			dataSetList.at(i)->deleteRep();
			baseSurvey->removeDataset(dataSetList.at(i));
			break;
		}
	}
}


QFileInfoList DataSelectorDialog::getFiles(std::string path, QStringList ext)
{

	QFileInfoList list;

	for( const auto & entry : boost::filesystem::directory_iterator(path))
	{
		for(int i=0;i<ext.size();i++)
		{
			if ( endsWith(entry.path().c_str(), ext[i].toStdString()) == 1)
			{
				if(boost::filesystem::is_regular_file(entry.path().c_str()) )
				{
					list.append(QFileInfo(QString::fromStdString(entry.path().c_str())));
				}
			}
		}

	}
	return list;
}

void DataSelectorDialog::filterHorizons(std::vector<QString>& horizonNames,
		std::vector<QString>& horizonPaths, std::vector<QString>& horizonExtractionDataPath,QString seismicDirPath) {
	// get extraction name
	std::vector<QString> horizonExtractionName;
	horizonExtractionName.resize(horizonNames.size(), QString());


	for (int i=0; i<horizonExtractionName.size(); i++) {
		// only keep horizons that match data
		bool isIJKFound = false;
		QFileInfo fileInfo(horizonPaths[i]);
		QDir dir = fileInfo.dir(); // object to find directory ImportExport
		QString previousDirName; // object to find folder IJK
		QString seismicDirName; // object to store which folder in IJK is used

		while (!isIJKFound && !dir.isRoot()) {
			if (dir.dirName().compare("ImportExport")==0 && previousDirName.compare("IJK")==0) {
				isIJKFound = true;
			} else {
				seismicDirName = previousDirName;
				previousDirName = dir.dirName();
				dir.cdUp();
			}
		}
		// if horizon not in ijk then we cannot use it because of extraction issues
		if (isIJKFound && !seismicDirName.isNull() && !seismicDirName.isEmpty()) {
			horizonExtractionName[i] = seismicDirName;
		}
	}


	// find extraction dataset
	std::vector<bool> isExtractionFoundArray;
	//QString seismicDirPath = m_selectorWidget->get_seismic_path0();
	QDir dir(seismicDirPath);

	dir.setFilter(QDir::Files);
	dir.setSorting(QDir::Name);

	QStringList filters;
	filters << "*.xt" << "*.cwt" ;
	//dir.setNameFilters(filters);

	QFileInfoList list =getFiles(seismicDirPath.toStdString(),filters); //dir.entryInfoList(filters);

	int N = list.size();
	std::vector<QString> seismic_list;
	seismic_list.resize(N);

	for (int i=0; i<list.size(); i++)
	{

		QFileInfo fileInfo = list.at(i);
		QString filename = fileInfo.fileName();
		seismic_list[i] = filename;
		//        fprintf(stderr, "name: %s -> %s\n", path.toStdString().c_str(), filename.toStdString().c_str());
	}



	horizonExtractionDataPath.resize(horizonNames.size(), QString());
	isExtractionFoundArray.resize(horizonNames.size(), false);

	int remainingToFind = horizonNames.size();
	std::size_t indexSeismic = 0;
	while(indexSeismic<seismic_list.size() && remainingToFind>0) {
		QString fullName = seismic_list[indexSeismic];
		QString sismageName = getSismageNameFromSeismicFile(seismicDirPath + fullName);

		for (int i=0; i<horizonNames.size(); i++) {
			if (!isExtractionFoundArray[i] && horizonExtractionName[i].compare(sismageName)==0) {
				isExtractionFoundArray[i] = true;
				horizonExtractionDataPath[i] = seismicDirPath + fullName;
				remainingToFind --;
			}
		}
		indexSeismic ++;
	}


	if (remainingToFind>0) {
		std::vector<QString> cleanedHorizonPaths;
		std::vector<QString> cleanedHorizonNames;
		std::vector<QString> cleanedHorizonExtractionDataPath;
		std::size_t Ncleaned = horizonPaths.size() - remainingToFind;
		cleanedHorizonPaths.resize(Ncleaned);
		cleanedHorizonNames.resize(Ncleaned);
		cleanedHorizonExtractionDataPath.resize(Ncleaned);
		std::size_t cleanedIndex = 0;
		for (int i=0; i<horizonNames.size(); i++) {
			if (isExtractionFoundArray[i]) {
				cleanedHorizonPaths[cleanedIndex] = horizonPaths[i];
				const QLatin1String LAST_PATH_IN_SETTINGS("mainwindow/lastPath");
				cleanedHorizonNames[cleanedIndex] = horizonNames[i];
				cleanedHorizonExtractionDataPath[cleanedIndex] = horizonExtractionDataPath[i];
				cleanedIndex ++;
			}
		}
		horizonNames = cleanedHorizonNames;
		horizonPaths = cleanedHorizonPaths;
		horizonExtractionDataPath = cleanedHorizonExtractionDataPath;
	}
}

SeismicSurvey* DataSelectorDialog::dataGetBaseSurvey(WorkingSetManager *pWorkingSetManager,QString surveyName,QString surveyPath,bool &rbIsNewSurvey){

	SeismicSurvey* baseSurvey = nullptr;
	//bool isNewSurvey = false;
	std::size_t idxSurvey = 0;
	QList<IData*> surveyList = pWorkingSetManager->folders().seismics->data();
	while (baseSurvey == nullptr && idxSurvey < surveyList.size()) {
		if (SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(surveyList[idxSurvey])) {
			if (survey->isIdPathIdentical(surveyPath)) {
				baseSurvey = survey;
			}
		}
		idxSurvey++;
	}
	rbIsNewSurvey = baseSurvey == nullptr;
	if (baseSurvey==nullptr) {
		SmSurvey3D survey(surveyPath.toStdString());
		if (survey.isValid()) {
			baseSurvey = new SeismicSurvey(pWorkingSetManager, surveyName,survey.inlineDim(), survey.xlineDim(), surveyPath);

			baseSurvey->setInlineXlineToXYTransfo(survey.inlineXlineToXYTransfo());
			baseSurvey->setIJToXYTransfo(survey.ijToXYTransfo());
		}
	}

	return baseSurvey;
}

void DataSelectorDialog::createSeismic(SeismicSurvey* baseSurvey,WorkingSetManager *pWorkingSetManager,const std::vector<QString>& datasetPaths,const std::vector<QString>& datasetNames,bool &rbIsNewSurvey,bool bUpdateDataSet){
	QList<Seismic3DAbstractDataset*> datasets = baseSurvey->datasets();
	std::size_t idDataset = 0;
	bool bIsDatasetLoaded = false;

	if(bUpdateDataSet == true){
		while (idDataset < datasets.size()) {
			bIsDatasetLoaded = false;
			QString dname = datasets[idDataset]->name();

			for (int i = 0; i < datasetPaths.size(); i++) {
				QString datasetPath = datasetPaths[i];
				QString datasetName = datasetNames[i];

				if(datasetName == dname){
					if (Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(datasets[idDataset])) {
						bIsDatasetLoaded = dataset->isIdPathIdentical(datasetPath);
						break;
					}
				}
			}

			if(bIsDatasetLoaded == false){
				DataSelectorDialog::removeDataset(baseSurvey,dname);
			}

			idDataset++;
		}
	}

	for (int i = 0; i < datasetPaths.size(); i++) {
		QString datasetName = datasetNames[i];
		QString datasetPath = datasetPaths[i];

		qDebug() << "Found dataset" << datasetName << datasetPaths.size();

		// search if data there
		bool isDatasetLoaded = false;
		std::size_t idxDataset = 0;
		QList<Seismic3DAbstractDataset*> datasets = baseSurvey->datasets();

		while (!isDatasetLoaded && idxDataset<datasets.size()) {

			if (Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(datasets[idxDataset])) {
				isDatasetLoaded = dataset->isIdPathIdentical(datasetPath);
				qDebug()<< datasetName << " : " << dataset->name();
			}
			idxDataset++;
		}

		if (isDatasetLoaded == false) {
			Seismic3DAbstractDataset::CUBE_TYPE type =	Seismic3DAbstractDataset::Seismic;
			if (datasetName.contains("rgt"))
				type = Seismic3DAbstractDataset::CUBE_TYPE::RGT;

			Seismic3DAbstractDataset *dataset = DataSelectorDialog::appendDataset(pWorkingSetManager,baseSurvey,datasetPath, datasetName, type, true);
		}
	}
	if (rbIsNewSurvey) {
		pWorkingSetManager->addSeismicSurvey(baseSurvey);
	}

}

void DataSelectorDialog::addHorizon(WorkingSetManager *pWorkingSetManager,std::vector<QString> horizonPaths ,std::vector<QString> horizonNames,QString seismicDirPath){
	std::vector<QString> horizonExtractionDataPath;

	DataSelectorDialog::filterHorizons(horizonNames, horizonPaths, horizonExtractionDataPath,seismicDirPath);
	 ;
	QList<IJKHorizon*> horizons = pWorkingSetManager->listIJKHorizons();

	for (long horizonIdx=0; horizonIdx<horizonNames.size(); horizonIdx++) {

		bool isHorizonLoaded = false;
		std::size_t idxHorizon = 0;
		while (!isHorizonLoaded && idxHorizon<horizons.size()) {

			isHorizonLoaded = horizons[idxHorizon]->isIdPathIdentical(horizonPaths[horizonIdx]);
			idxHorizon++;
		}
		if (!isHorizonLoaded) {

			IJKHorizon* horizon = new IJKHorizon(horizonNames[horizonIdx], horizonPaths[horizonIdx],horizonExtractionDataPath[horizonIdx], pWorkingSetManager);
			pWorkingSetManager->addIJKHorizon(horizon);
		}
	}

	//	m_manager->set_horizons(horizonNames, horizonPaths, horizonExtractionDataPath);
}

void DataSelectorDialog::addWellBore(WorkingSetManager *pWorkingSetManager,const std::vector<WELLLIST>& wells,
		const std::vector<MARKER>& wellPicksList, bool bUpdateWellBore){

//	steady_clock::time_point debut = steady_clock::now();

	removeUnselectedWells(pWorkingSetManager,wells,bUpdateWellBore);
	int index=0;
	for (WELLLIST well : wells) {
		long index = 0;
	//	steady_clock::time_point debut = steady_clock::now();
		bool isValid = addWellHead(well,pWorkingSetManager,index);
		if (isValid) {
			for (WELLBORELIST wellbore : well.wellborelist) {
				WellBore *bore = createUpdatebore(wellbore,index,pWorkingSetManager);
				if (bore!=nullptr) {
					if(bUpdateWellBore == true){
					   deleteUnselectedPick(bore,wellPicksList,well.head_fullname,wellbore.bore_fullname);
					}
					createPicks(pWorkingSetManager,wellPicksList,well.head_fullname,wellbore.bore_fullname,bore);// create picks
				}
			}
		}
	//	steady_clock::time_point fin = steady_clock::now();
	//	duration<double> time5= duration_cast<duration<double>>(fin - debut);


	}

}


void DataSelectorDialog::addNurbs(WorkingSetManager *pWorkingSetManager ,std::vector<QString> nurbsPath,std::vector<QString> nurbsNames)
{

	for(int i=0;i<nurbsNames.size();i++)
	{
		pWorkingSetManager->addNurbs(nurbsPath[i],nurbsNames[i]);
	}
}



bool DataSelectorDialog::addWellHead(const WELLLIST& well,WorkingSetManager *pWorkingSetManager,long &rIndex){
	bool isValid = false;
	QDir wellHeadDir(well.head_fullname);
	QString descFile;
	QStringList descFiles = wellHeadDir.entryList(QStringList() << "*.desc", QDir::Files);
	QList<WellHead*> wellHeads = pWorkingSetManager->listWellHead();

	isValid = descFiles.count()>0 && (well.wellborelist.size()!=0);
	if (isValid) {
		descFile = descFiles.first();
	}
	if (isValid) {
		while (rIndex<wellHeads.size() && !wellHeads[rIndex]->isIdPathIdentical(wellHeadDir.absoluteFilePath(descFile))) {
			rIndex++;
		}
		if (rIndex >= wellHeads.size()) {
			WellHead* newWellHead = WellHead::getWellHeadFromDescFile(wellHeadDir.absoluteFilePath(descFile), pWorkingSetManager);
			isValid = newWellHead != nullptr;
			if (isValid) {
				wellHeads.push_back(newWellHead);
				pWorkingSetManager->addWellHead(newWellHead);

				newWellHead->setAllDisplayPreference(true);
			}
		}
	}

	return isValid;
}

void DataSelectorDialog::removeUnselectedWells(WorkingSetManager *pWorkingSetManager,const std::vector<WELLLIST>& wells,bool bUpdateWellBore){
	// MZR 20082021
	if(bUpdateWellBore == true){
		QList<WellHead*> wellHeads = pWorkingSetManager->listWellHead();
		int i=0;
		for(WellHead* pWell:wellHeads){
			bool bIsWellLoaded =false;

			// WellHead idPath point to the well head desc file
			// head_fullname point to well head folder (no "/" at the end)
			// To compare both we recover the well head folder from the desc location
			QString pWellFullName = QFileInfo(pWell->idPath()).dir().absolutePath();
			for(i=0;i<wells.size();i++){
				WELLLIST wellist= wells[i];
				if((wellist.head_fullname == pWellFullName) && (wellist.wellborelist.size()!=0)){
					bIsWellLoaded = true;
					break;
				}
			}
			if(bIsWellLoaded == false){
				QList<WellBore*> wellBore = pWell->wellBores();
				for(WellBore* pWellBore:wellBore){
					std::vector<MARKER> wellPicksList;
					// 16 12 2022 : the paths below are not used by the function deleteUnselectedPick
					QString well_head_fullname = QFileInfo(pWellBore->wellHead()->idPath()).dir().absolutePath();
					QString well_bore_fullname = QFileInfo(pWellBore->idPath()).dir().absolutePath();
					deleteUnselectedPick(pWellBore,wellPicksList,well_head_fullname,well_bore_fullname);

					pWellBore->deleteRep();
					break;
				}
				pWorkingSetManager->removeWellHead(pWell);
			}
		}
	}
}

WellBore *DataSelectorDialog::createUpdatebore(const WELLBORELIST& wellbore,long index,WorkingSetManager *pWorkingSetManager){
//	qDebug() << "Try well : " << wellbore.bore_tinyname ;
	WellBore* bore = nullptr;
	bool isWellBoreLoaded = false;

	// get well head
	QList<WellHead*> wellHeads = pWorkingSetManager->listWellHead();
	WellHead* head;
	if (index < wellHeads.size()) {
		head = wellHeads[index];
	} else {
		head = wellHeads[wellHeads.size()-1];
	}
	QList<WellBore*> wellBoresLoaded = head->wellBores();

	// create well bore
	QDir wellBoreDir(wellbore.bore_fullname);
	QStringList descFiles = wellBoreDir.entryList(QStringList() << "*.desc", QDir::Files);
	long indexWellBoreLoaded = 0;
	while (indexWellBoreLoaded<wellBoresLoaded.count() && !isWellBoreLoaded){
		isWellBoreLoaded = wellBoresLoaded[indexWellBoreLoaded]->isIdPathIdentical(wellBoreDir.absoluteFilePath(descFiles[0]));
		if (isWellBoreLoaded){
			bore = wellBoresLoaded[indexWellBoreLoaded];
		}
		indexWellBoreLoaded++;
	}

	if (descFiles.size()>0 && !isWellBoreLoaded){
		QString wellBoreDescFile = descFiles[0];
		bore = new WellBore(pWorkingSetManager, wellBoreDir.absoluteFilePath(wellBoreDescFile),wellbore.deviation_fullname, wellbore.tf2p_fullname, wellbore.tf2p_tinyname,wellbore.log_fullname, wellbore.log_tinyname, head);
	}else{
		// MZR 05082021
		if(bore != nullptr){
			bool resultCmplog = true;
			bool resultCmptf2p = true;

			std::vector<QString> logVect = bore->logsNames();
			std::vector<QString> TfpVect = bore->tfpsNames();

			if(TfpVect.size() == wellbore.tf2p_tinyname.size()){
				if (wellbore.tf2p_tinyname.size()!= 0){
					resultCmptf2p = std::equal(wellbore.tf2p_tinyname.begin(), wellbore.tf2p_tinyname.end(), TfpVect.begin(),[](QString str1, QString str2)
							{
						return !str1.compare(str2);
							});
				}
			}else{
				resultCmptf2p = false;
			}

			if(logVect.size() == wellbore.log_tinyname.size()){
				if (wellbore.log_tinyname.size()!= 0){
					resultCmplog = std::equal(wellbore.log_tinyname.begin(), wellbore.log_tinyname.end(),logVect.begin(),[](QString str1, QString str2)
							{
						return !str1.compare(str2);
							});
				}
			}else{
				resultCmplog = false;
			}

			if(resultCmplog == false || resultCmptf2p == false){
				bore->SetTfpName(wellbore.tf2p_tinyname);
				bore->SetTfpsPath(wellbore.tf2p_fullname);
				bore->SetlogName(wellbore.log_tinyname);
				bore->SetlogPath(wellbore.log_fullname);
			}
		}
	}

	if (!isWellBoreLoaded) {
		head->addWellBore(bore);

		bore->setAllDisplayPreference(true);
	}

	return bore;
}

void DataSelectorDialog::deleteUnselectedPick(WellBore* bore,const std::vector<MARKER>& pickList,
		QString wellHeadPath,QString wellBorePath)
{
	QList<WellPick*> picksLoaded = bore->picks();
	long pickIndex = 0;
	bool pickLoaded = false;

	while (pickIndex < picksLoaded.size()) {
		pickLoaded = false;
		WellPick* pick = picksLoaded[pickIndex];
		QString pickFile;

		bool pickNotFound = true;
		// try all picks, all well head and all well bore
		int idxPick = 0;
		while (pickNotFound && idxPick<pickList.size()) {
			int idxHead = 0;
			while (pickNotFound && idxHead<pickList[idxPick].wellPickLists.size()) {
				const WELLPICKSLIST& wellBore = pickList[idxPick].wellPickLists[idxHead];
				int idxBore = 0;
				while (pickNotFound && idxBore<wellBore.wellBore.size()) {
					pickNotFound = !pick->isIdPathIdentical(wellBore.wellBore[idxBore].picksPath);
					if (pickNotFound) {
						idxBore++;
					}
				}

				if (pickNotFound) {
					idxHead++;
				}
			}

			if (pickNotFound) {
				idxPick++;
			}
		}
//		for (std::size_t idxPick = 0; idxPick < wellbore.picks_fullname.size(); idxPick++) {
//			pickFile = wellbore.picks_fullname[idxPick];
//			pickLoaded = pick->isIdPathIdentical(pickFile);
//			if(pickLoaded == true){
//				break;
//			}
//		}
		pickLoaded = !pickNotFound;
		if(pickLoaded == false){
			bore->removePick(pick);
			Marker* marker = pick->currentMarker();
			if (marker) {
				marker->removeWellPick(pick);
			}
			pick->deleteLater();
			if (marker && marker->wellPicks().size()==0) {
				WorkingSetManager* manager = marker->workingSetManager();
				if (manager) {
					manager->removeMarker(marker);
				} else {
					marker->deleteLater();
				}
			}
			//m_manager->removeMarker(pick->currentMarker());
		}
		pickIndex++;
	}
}

void DataSelectorDialog::createPicks(WorkingSetManager *pWorkingSetManager,const std::vector<MARKER>& markers,
		QString wellHeadPath,QString wellBorePath,WellBore* bore)
{
	bool pickLoaded = false;
	long pickIndex = 0;
	QList<WellPick*> picksLoaded = bore->picks();

	// create picks
	for (std::size_t idxPick=0; idxPick<markers.size(); idxPick++) {
		// search if pick available for this well
		const std::vector<WELLPICKSLIST>& wellHeadsForPick = markers[idxPick].wellPickLists;
		bool pickNotAvailable = true;
		int tryWellHeadIdx = 0;
		while (pickNotAvailable && tryWellHeadIdx<wellHeadsForPick.size()) {
			pickNotAvailable = wellHeadPath.compare(wellHeadsForPick[tryWellHeadIdx].path)!=0;
			if (pickNotAvailable) {
				tryWellHeadIdx++;
			}
		}
		int tryWellBoreIdx = 0;
		if (!pickNotAvailable) {
			// search well bore
			pickNotAvailable = true;
			const WELLPICKSLIST& wellBoresForPick = wellHeadsForPick[tryWellHeadIdx];
			while (pickNotAvailable && tryWellBoreIdx<wellBoresForPick.wellBore.size()) {
				pickNotAvailable = wellBorePath.compare(wellBoresForPick.wellBore[tryWellBoreIdx].borePath)!=0;
				if (pickNotAvailable) {
					tryWellBoreIdx++;
				}
			}
		}

		if (!pickNotAvailable) {
			const WELLBOREPICKSLIST& pick = wellHeadsForPick[tryWellHeadIdx].wellBore[tryWellBoreIdx];
			QString pickFile = pick.picksPath;
			pickLoaded = false;
			pickIndex = 0;
			//qDebug() << "Try pick : " << pickFile;
			while (pickIndex<picksLoaded.size() && !pickLoaded) {
				pickLoaded = picksLoaded[pickIndex]->isIdPathIdentical(pickFile);
				pickIndex++;
			}
			if (!pickLoaded) {
				WellPick* pick = WellPick::getWellPickFromDescFile(bore, markers[idxPick].color, pickFile, pWorkingSetManager);
				if (pick!=nullptr) {
					bore->addPick(pick);

					pick->setAllDisplayPreference(true);
				}
			}
		}
	}
}

void DataSelectorDialog::accepted() {
	if (!m_acceptMutex.try_lock()) {
		return;
	}
	//   QSettings settings;
	//   settings.setValue(RGT_SEISMIC_SLICER_SURVEY_PATH, surveyPath);

	//steady_clock::time_point start = steady_clock::now();
	if ( !m_selectorWidget ) { accept(); return; }
	m_selectorWidget->fill_empty_logs_list();


	std::vector<QString> datasetNames = m_selectorWidget->get_seismic_names();
	std::vector<QString> horizonPaths = m_selectorWidget->get_horizon_fullpath_names();
	std::vector<QString> horizonNames = m_selectorWidget->get_horizon_names();
	std::vector<QString> datasetPaths = m_selectorWidget->get_seismic_fullpath_names();

	std::vector<QString> nurbsNames = m_selectorWidget->get_nurbs_names();
	std::vector<QString> nurbspath = m_selectorWidget->get_nurbs_fullnames();

	std::vector<QString> freeHorizonNames = m_selectorWidget->get_freehorizon_names_basket();
	std::vector<QString> freeHorizonPaths = m_selectorWidget->get_freehorizon_fullnames_basket();
	std::vector<QString> isoHorizonNames = m_selectorWidget->get_isohorizon_names_basket();
	std::vector<QString> isoHorizonPaths = m_selectorWidget->get_isohorizon_fullnames_basket();
	std::vector<QString> horizonAnimNames = m_selectorWidget->get_horizonanim_names_basket();
	std::vector<QString> horizonAnimPaths = m_selectorWidget->get_horizonanim_fullnames_basket();


//	steady_clock::time_point etape1 = steady_clock::now();
	std::vector<WELLLIST> wells = m_selectorWidget->get_well_list();
//	steady_clock::time_point etape2 = steady_clock::now();
	QString surveyPath = m_selectorWidget->get_survey_fullpath_name();
	QString surveyName = m_selectorWidget->get_survey_name();
	QString seismicDirPath = m_selectorWidget->get_seismic_path0();
	QFileInfo surveyInfo(surveyPath);

//	steady_clock::time_point etape3 = steady_clock::now();
	qDebug() << "Found survey" << surveyName << " " << surveyPath;
/*	duration<double> time1= duration_cast<duration<double>>(etape1 - start);
	duration<double> time2= duration_cast<duration<double>>(etape2 - etape1);
	duration<double> time3= duration_cast<duration<double>>(etape3 - etape2);

	qDebug() <<" etape1:" << time1.count() << " seconds.";
	qDebug() <<" etape2:" << time2.count() << " seconds.";
	qDebug() <<" etape3:" << time3.count() << " seconds.";*/
	// if ((surveyInfo.isDir() == true) && (datasetPaths.empty() == false))
	if ((surveyInfo.isDir() == true) )
	{
		bool isNewSurvey = false;


		SeismicSurvey* baseSurvey = DataSelectorDialog::dataGetBaseSurvey(m_manager,surveyName,surveyPath,isNewSurvey);
		if(baseSurvey != nullptr){
			//steady_clock::time_point etape4 = steady_clock::now();
			DataSelectorDialog::createSeismic(baseSurvey,m_manager,datasetPaths,datasetNames,isNewSurvey);

			//steady_clock::time_point etape5 = steady_clock::now();
			DataSelectorDialog::addHorizon(m_manager,horizonPaths,horizonNames,seismicDirPath);
			//steady_clock::time_point etape6 = steady_clock::now();

			DataSelectorDialog::addWellBore(m_manager,wells,m_selectorWidget->getPicksSortedWells());


			DataSelectorDialog::addNurbs(m_manager,nurbspath,nurbsNames);

			DataSelectorDialog::addNVHorizons(m_manager,baseSurvey,freeHorizonPaths,freeHorizonNames);

			DataSelectorDialog::addNVIsoHorizons(m_manager,baseSurvey,isoHorizonPaths,isoHorizonNames);

			DataSelectorDialog::addNVHorizonAnims(m_manager,baseSurvey,horizonAnimPaths,horizonAnimNames);
			/*steady_clock::time_point etape7 = steady_clock::now();
			duration<double> time5= duration_cast<duration<double>>(etape5 - etape4);
			duration<double> time6= duration_cast<duration<double>>(etape6 - etape5);
			duration<double> time7= duration_cast<duration<double>>(etape7 - etape6);
			qDebug() <<" createSeismic:" << time5.count() << " seconds.";
			qDebug() <<" addHorizon:" << time6.count() << " seconds.";
			qDebug() <<" addWellBore:" << time7.count() << " seconds.";*/
		}


		for (int i=0; i<horizonNames.size(); i++)
			fprintf(stderr, "** %d %s %s\n", i, horizonNames[i].toStdString().c_str(), horizonPaths[i].toStdString().c_str());
	}

	//steady_clock::time_point end = steady_clock::now();
	//duration<double> time_spanTot = duration_cast<duration<double>>(end - start);



//	qDebug() <<" ACCEPT Total:" ;

	m_acceptMutex.unlock();

	accept();
}

bool DataSelectorDialog::addNVHorizons(WorkingSetManager *pWorkingSetManager, SeismicSurvey *survey, std::vector<QString> horizonPaths, std::vector<QString> horizonName)
{
	for(int i=0; i<horizonPaths.size(); i++)
	{
		if(!pWorkingSetManager->containsFreeHorizon(horizonPaths[i]))
		{
			FreeHorizon *freeHorizon = new FreeHorizon(pWorkingSetManager, survey, horizonPaths[i], horizonName[i]);
			pWorkingSetManager->addFreeHorizons(freeHorizon);
			IData* isochronData = freeHorizon->getIsochronData();
			if (isochronData) {
				isochronData->setDisplayPreferences({ViewType::InlineView, ViewType::XLineView, ViewType::RandomView}, true);
			}
		}
		else
		{
			FreeHorizon *freeHorizon = pWorkingSetManager->getFreeHorizon(horizonPaths[i]);
			std::vector<QString> list = FreeHorizonQManager::getAttributData(horizonPaths[i]);
			for (int j=0; j<list.size(); j++)
			{
				if ( freeHorizon->isAttributExists(list[j]) ) continue;
				freeHorizon->freeHorizonAttributCreate(list[j]);
			}
			qDebug() << "exist";
		}
	}
	return true;
}

bool DataSelectorDialog::removeNVHorizonsAttribut(WorkingSetManager *pWorkingSetManager, SeismicSurvey *survey,
		QString horizonPath, QString horizonName,
		QString attributPath, QString attributName)
{
	if(pWorkingSetManager->containsFreeHorizon(horizonPath))
	{
		FreeHorizon *freeHorizon = pWorkingSetManager->getFreeHorizon(horizonPath);
		if ( freeHorizon == nullptr ) return false;
		if ( freeHorizon->isAttributExists(attributName) )
			freeHorizon->freeHorizonAttributRemove(attributName);
	}
	return true;
}


bool DataSelectorDialog::addNVIsoHorizons(WorkingSetManager *pWorkingSetManager, SeismicSurvey *survey, std::vector<QString> horizonPaths, std::vector<QString> horizonName)
{
	for(int i=0; i<horizonPaths.size(); i++)
	{
		IsoHorizon *isoHorizon = new IsoHorizon(pWorkingSetManager, survey, horizonPaths[i], horizonName[i]);
		pWorkingSetManager->addIsoHorizons(isoHorizon);
		IData* isochronData = isoHorizon->getIsochronData();
		if (isochronData) {
			isochronData->setDisplayPreferences({ViewType::InlineView, ViewType::XLineView, ViewType::RandomView}, true);
		}
	}
	return true;
}

bool DataSelectorDialog::addNVHorizonAnims(WorkingSetManager *pWorkingSetManager, SeismicSurvey *survey, std::vector<QString> animPaths, std::vector<QString> animNames)
{
	for(int i=0; i<animPaths.size(); i++)
	{
		bool valid = !pWorkingSetManager->containsHorizonAnim(animNames[i]);
		HorizonDataRep::HorizonAnimParams params;
		if (valid)
		{
			params = HorizonDataRep::readAnimationHorizon(animPaths[i], &valid);
		}

		if (valid)
		{
			// load horizons
			std::vector<QString> listepath;
			std::vector<QString> listename;
			for(int i=0;i<params.horizons.size();i++)
			{
				QFileInfo fileinfo(params.horizons[i]);
				listepath.push_back(params.horizons[i]);
				listename.push_back(fileinfo.baseName());
			}
			DataSelectorDialog::addNVHorizons(pWorkingSetManager, survey, listepath, listename);

			HorizonFolderData* horizonFolderData = new HorizonFolderData(pWorkingSetManager,animNames[i], params.horizons);

			pWorkingSetManager->addHorizonAnimData(horizonFolderData);
			horizonFolderData->setDisplayPreferences({InlineView,XLineView,RandomView},true);
		}
	}
	return true;
}

void DataSelectorDialog::loadSession() {
	m_selectorWidget->load_session_gui();
}

void DataSelectorDialog::saveSession() {
	m_selectorWidget->save_session_gui();
}

// code to transfert tarum session to next vison session
// ! warning does not worry about overwriting files !

//void DataSelectorDialog::run() {
//	QDir dir("/data/PLI/naamen/DEMO_Geotime_Tarum/");
//	QFileInfoList txtList = dir.entryInfoList(QStringList() << "*.txt", QDir::Files);
//
//	for (QFileInfo txtFileInfo : txtList) {
//		QString jsonFilePath = "/data/PLI/NextVision/sessions/" + txtFileInfo.owner() + "/" + txtFileInfo.baseName() + ".json";
//		QString txtFilePath = txtFileInfo.absoluteFilePath();
//
//		// mkdir
//		if (!QFileInfo(jsonFilePath).absoluteDir().exists()) {
//			QDir dir(QFileInfo(jsonFilePath).absoluteDir());
//			QString name = dir.dirName();
//			dir.cdUp();
//			dir.mkdir(name);
//		}
//
//		if (txtFilePath.compare("/data/PLI/naamen/DEMO_Geotime_Tarum/UMC-NK_as.txt")==0) {
//			qDebug() << "breakpoint";
//		}
//
//		QFile file(txtFilePath);
//		if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
//			continue;
//
//		QString projectType;
//		QString project;
//		QString survey;
//		QStringList seismics;
//
//	    QTextStream in(&file);
//	    bool doContinue = false;
//	    while (!in.atEnd()) {
//	        QString line = in.readLine();
//	        QStringList lineSplit = line.split(" ");
//	        if (lineSplit.count()<2) {
//	        	continue;
//	        }
//	        line = lineSplit[1]; // remove name
//
//	        // get seismic sismage name
//	        QString seismicName = getSismageNameFromSeismicFile(line);
//	        if (QFileInfo(line).suffix().compare("cwt")==0) {
//	        	seismicName += QString(" (compress)");
//	        }
//
//	        // init project infos
//	        if (projectType.isNull() || projectType.isEmpty()) {
//	        	QDir searchDir = QFileInfo(line).absoluteDir();
//	        	bool valid = searchDir.cdUp();
//	        	valid = searchDir.cdUp();
//	        	survey = searchDir.dirName(); // path survey name
//
//	        	// search desc file
//	        	QStringList descList = searchDir.entryList(QStringList() << "*.desc", QDir::Files);
//
//	        	if (descList.size()>0) {
//					QFile fileSurvey(searchDir.absoluteFilePath(descList.first()));
//					if (!fileSurvey.open(QIODevice::ReadOnly | QIODevice::Text)) {
//						QTextStream inSurvey(&fileSurvey);
////						void DataSelectorDialog::run() {
////							QDir dir("/data/PLI/naamen/DEMO_Geotime_Tarum/");
////							QFileInfoList txtList = dir.entryInfoList(QStringList() << "*.txt", QDir::Files);
////
////							for (QFileInfo txtFileInfo : txtList) {
////								QString jsonFilePath = "/data/PLI/NextVision/sessions/" + txtFileInfo.owner() + "/" + txtFileInfo.baseName() + ".json";
////								QString txtFilePath = txtFileInfo.absoluteFilePath();
////
////								// mkdir
////								if (!QFileInfo(jsonFilePath).absoluteDir().exists()) {
////									QDir dir(QFileInfo(jsonFilePath).absoluteDir());
////									QString name = dir.dirName();
////									dir.cdUp();
////									dir.mkdir(name);
////								}
////
////								if (txtFilePath.compare("/data/PLI/naamen/DEMO_Geotime_Tarum/UMC-NK_as.txt")==0) {
////									qDebug() << "breakpoint";
////								}
////
////								QFile file(txtFilePath);
////								if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
////									continue;
////
////								QString projectType;
////								QString project;
////								QString survey;
////								QStringList seismics;
////
////							    QTextStream in(&file);
////							    bool doContinue = false;
////							    while (!in.atEnd()) {
////							        QString line = in.readLine();
//						bool nameFound = false;
//						while (!inSurvey.atEnd() && !nameFound) {
//							QString lineSurvey = in.readLine();
//							QStringList split = lineSurvey.split("=");
//							nameFound = split.first().compare("name")==0 && split.count()==2 && !split.last().isNull() && !split.last().isEmpty();
//							if (nameFound){
//								survey = split.last();
//							}
//						}
//					}
//	        	}
//
//	        	valid = searchDir.cdUp();
//	    	    valid = searchDir.cdUp();
//	        	valid = searchDir.cdUp();
//	        	project = searchDir.dirName(); // path survey name
//
//	        	valid = searchDir.cdUp();
//	        	QString dirName = searchDir.path();
//	        	/* if (dirName.compare("/data/sismage/IMA3G/DIR_PROJET")==0) {
//	        		projectType = "R&D";
//	        	} else */if (dirName.compare("/data/IMA3G/DIR_PROJET")==0) {
//	        		projectType = "PROD";
//	        	}/* else if (dirName.compare("/data/PLI/DIR_PROJET")==0) {
//	        		projectType = "PLI";
//	        	}*/ else {
//	        		// TODO
//	        		qDebug() << "rejected" << txtFilePath;
//	        		doContinue = true;
//	        		break;
//	        	}
//	        	qDebug() <<  "accepted" << txtFilePath;
//
//	        }
//
//	        seismics.append(seismicName);
//	    }
//
//	    if (doContinue) {
//	    	continue;
//	    }
//
//	    if (!QFileInfo(jsonFilePath).exists()) {
//			QFile fileJson(jsonFilePath);
//			if (!fileJson.open(QIODevice::WriteOnly))
//				return;
//
//			QJsonObject obj;
//			obj.insert(projectTypeKey, projectType);
//			obj.insert(projectKey, project);
//			obj.insert(surveyKey, survey);
//
//			QJsonArray seismicsArray;
//			for (std::size_t arrayIdx=0; arrayIdx<seismics.count(); arrayIdx++) {
//				seismicsArray.append(seismics[arrayIdx]);
//			}
//			obj.insert(seismicKey, seismicsArray);
//
//			QJsonDocument doc(obj);
//			fileJson.write(doc.toJson());
//	    } else {
//	    	qDebug() << "file " << jsonFilePath << "exist";
//	    }
//	}
//}
