
#include <string>
#include <iostream>

#include <qboxlayout.h>
#include <QLineEdit>
#include <QWidget>

#include <qlabel.h>
#include <qpushbutton.h>
#include <qgroupbox.h>
#include <QFormLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QMainWindow>
#include <QDockWidget>
#include <QScrollArea>
#include <QInputDialog>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QProgressBar>
#include <QListWidget>
#include <QFormLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QImage>
#include <QIcon>
#include <QApplication>
#include <QGraphicsScene>
#include <QStack>

#include "Xt.h"
#include "DataSelectorDialog.h"
#include "stringselectordialog.h"
#include "dialog/validator/SimpleDoubleValidator.h"
#include "dialog/validator/OutlinedQLineEdit.h"
#include "abstractsectionview.h"
#include "basemapview.h"
#include "stackbasemapview.h"
#include "basemapgraphicsview.h"
#include "randomlineview.h"
#include "slicer/data/layerslice/LayerSlice.h"
#include "slicer/data/layerslice/rgblayerslice.h"
#include "monotypegraphicsview.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "seismicsurvey.h"
#include "slicerep.h"
#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "workingsetmanager.h"

#include "sliceutils.h"
#include "basemapview.h"
#include "basemapgraphicsview.h"
#include "multiseedrgt.h"
#include "multiseedslicerep.h"
#include "multiseedrandomrep.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "geotimegraphicsview.h"
#include "qgraphicsreptreewidgetitem.h"
#include "qinnerviewtreewidgetitem.h"
#include "viewqt3d.h"
#include "fixedlayerfromdataset.h"
#include "fixedlayerfromdatasetrep.h"
#include "SeismicPropagator.h"
#include "LayerRGTInterpolator.h"
#include "ijkhorizon.h"
#include "savehorizondialog.h"
#include "selectseedfrommarkers.h"
#include "folderdata.h"
#include "LayerSpectrumDialog.h"
#include "GeotimeMarkerQCDialog.h"
#include "RgtVolumicDialog.h"
#include "fileio2.h"
#include <surface_stack.h>
// #include <rgtProcessing.h>

#include "cultural.h"

//#include "viewers/view2d/ViewHorizonExtension.h"
//#include "viewers/view2d/MultiSeedHorizonExtension.h"
//#include "viewers/view2d/visual/DatasetSyncViewer2dVisual.h"
//#include "viewers/geotimeview/GeotimeView.h"
//#include "LayerSpectrumProcess.h"

//#include "itkImageFileReader.h"
//#include "itkImageFileWriter.h"

//#include "CubeIO.h"
//#include "CubeIOException.h"
//#include "Cube.h"

#include <limits>
#include <png.h>
#include <ihm.h>



//const char * LayerSpectrumDialog::EXT_STR = "_Converted";

RgtVolumicDialog::RgtVolumicDialog(
		Seismic3DAbstractDataset *datasetS, int channelS, Seismic3DAbstractDataset *datasetT, int channelT,
		GeotimeGraphicsView* viewer,
		QWidget *parent): QWidget(),
		m_datasetS( datasetS), m_datasetT( datasetT),
		m_originViewer (viewer),
		//				m_pickingTaskSection(Qt::LeftButton, this),
		m_pickingTaskMap(Qt::LeftButton, this),
		m_geotimeViewTrackerExtension({Qt::Key_U, Qt::Key_C, Qt::Key_P}, this)/*,m_originMainViewer(viewerMain)*/
	{

	/*
	this->setAttribute(Qt::WA_DeleteOnClose);
	m_channelS = channelS;
	m_channelT = channelT;
	constraintCounter = 0;
	initIhm();
	trt_startNewPicking();
	// constraintName.push_back("constraint 1");
	// std::vector<RgtSeed> tmp;
	// constraintPoints.push_back(tmp);
	// displayConstraintNames();
	// qListWidgetPicking->setCurrentRow(0);

	// MultiSeedRgt* m_rgt = nullptr;
	// m_rgt = new MultiSeedRgt("seed_horizon", m_datasetS->workingSetManager(),
	//			dynamic_cast<Seismic3DDataset*>(m_datasetS), m_channelS, dynamic_cast<Seismic3DDataset*>(m_datasetT), m_channelT);
	 *
	 */
}
RgtVolumicDialog::RgtVolumicDialog():
		m_pickingTaskMap(Qt::LeftButton, this),
		m_geotimeViewTrackerExtension({Qt::Key_U, Qt::Key_C, Qt::Key_P}, this)
{

}


RgtVolumicDialog::~RgtVolumicDialog() {

}


void RgtVolumicDialog::initIhm()
{
	/*
	QVBoxLayout * mainLayout00 = new QVBoxLayout(this);
	QPushButton *qPushButtonStartStopNewPicking = new QPushButton("Start new picking");
	qListWidgetPicking = new QListWidget();
	QPushButton *qPushButtonContinuePicking = new QPushButton("Continue existing picking");
	QPushButton *qPushButtonErasePicking = new QPushButton("Erase picking");
	QPushButton *qPushButtonDisplayConstaints = new QPushButton("Display Constraints");
	QPushButton *qPushButtonRunRgt = new QPushButton("Run");
	QPushButton *qPushButtonDebug = new QPushButton("D E B U G");


	mainLayout00->addWidget(qPushButtonStartStopNewPicking);
	mainLayout00->addWidget(qListWidgetPicking);
	// mainLayout00->addWidget(qPushButtonContinuePicking);
	mainLayout00->addWidget(qPushButtonErasePicking);
	mainLayout00->addWidget(qPushButtonDisplayConstaints);
	mainLayout00->addWidget(qPushButtonRunRgt);
	mainLayout00->addWidget(qPushButtonDebug);

	connect(qPushButtonDisplayConstaints, SIGNAL(clicked()), this, SLOT(trt_displayConstraints()));
	connect(qPushButtonStartStopNewPicking, SIGNAL(clicked()), this, SLOT(trt_startNewPicking()));
	connect(qPushButtonErasePicking, SIGNAL(clicked()), this, SLOT(trt_erasePicking()));
	connect(qPushButtonRunRgt, SIGNAL(clicked()), this, SLOT(trt_runRgt()));
	connect(qListWidgetPicking, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(trt_constraintListClick(QListWidgetItem*)));

	connect(qPushButtonDebug, SIGNAL(clicked()), this, SLOT(trt_debugClick()));


	setWindowTitle("RGT isovalues picking");
	*/
}

void RgtVolumicDialog::trt_displayConstraints()
{
	/*
	for (int i=0; i<constraintPoints.size(); i++)
	{
		std::vector<RgtSeed> seed = constraintPoints[i];
		for (int j=0; j<seed.size(); j++)
		{
			fprintf(stderr, "[%d:%d] X: %d \tY: %d \t %d\t %d\t%d\n", i, j, seed[j].x, seed[j].y, seed[j].z, seed[j].seismicValue, seed[j].rgtValue);
		}
	}
	*/
}

void RgtVolumicDialog::trt_startNewPicking()
{
	/*
	constraintName.push_back("constraint " + constraintCounter++);
	std::vector<RgtSeed> seed;
	constraintPoints.push_back(seed);
	displayConstraintNames();
	qListWidgetPicking->setCurrentRow(constraintName.size()-1);
	trt_constraintListClick(nullptr);
	*/
}

void RgtVolumicDialog::trt_erasePicking()
{
	/*
	int index = getConstraintSelectedIndex();
	if ( index < 0 ) return;
	constraintPoints.erase(constraintPoints.begin()+index);
	constraintName.erase(constraintName.begin()+index);
	displayConstraintNames();
	*/
}

int RgtVolumicDialog::trt_constraintListClick(QListWidgetItem* list)
{
	/*
	if ( m_layerSpectrumDialog )
		m_layerSpectrumDialog->clearHorizon();
	int pickingNo = getConstraintSelectedIndex();
	if ( pickingNo < 0 ) return 0;
	addPoints(constraintPoints[pickingNo]);
	*/
}

void RgtVolumicDialog::trt_runRgt()
{
	// rgtRun();
}

int RgtVolumicDialog::getConstraintSelectedIndex()
{
	/*
	QList<QListWidgetItem *> listItem = qListWidgetPicking->selectedItems();
	if ( listItem.empty() ) return -1;
	QString selectedName = listItem[0]->text();
	for (int i=0; i<constraintName.size(); i++)
	{
		if ( QString::compare(qListWidgetPicking->item(i)->text(), selectedName) == 0 ) return i;
	}
	*/
	return -1;
}

void RgtVolumicDialog::displayConstraintNames()
{
	/*
	qListWidgetPicking->clear();
	for (int i=0; i<constraintName.size(); i++)
	{
		QString name = QString::number(i+1) + " - " + constraintName[i] + " [size: " + QString::number(constraintPoints[i].size()) + " ]";
		qListWidgetPicking->addItem(name);
	}
	*/
}

void RgtVolumicDialog::setMultiSeedHorizon(MultiSeedHorizon *multiseedhorizon)
{
	// m_multiSeedHorizon = multiseedhorizon;
}

void RgtVolumicDialog::setLayerSpectrumDialog(LayerSpectrumDialog *layerSpectrumDialog)
{
	// m_layerSpectrumDialog = layerSpectrumDialog;
}

void RgtVolumicDialog::setLayerSpectrumDialog(MultiSeedSliceRep *multiSeedSliceRep)
{
	// m_multiSeedSliceRep = multiSeedSliceRep;
}

void RgtVolumicDialog::addPoints(std::vector<RgtSeed> seeds)
{
	/*
	if ( !m_layerSpectrumDialog || !m_multiSeedSliceRep || seeds.empty() ) return;
	m_layerSpectrumDialog->clearHorizon();
	for (RgtSeed seed:seeds)
		m_multiSeedSliceRep->addPoint(seed);
		*/
}


void RgtVolumicDialog::addPoint(RgtSeed seed)
{
	/*
	fprintf(stderr, "X: %d \tY: %d \t %d\t %d\t%d\n", seed.x, seed.y, seed.z, seed.seismicValue, seed.rgtValue);
	int pickingNo = getConstraintSelectedIndex();
	if ( pickingNo < 0)
	{
		QMessageBox msgBox;
		msgBox.setText("You must select a constraint first");
		msgBox.exec();
		return;
	}
	constraintPoints[pickingNo].push_back(seed);
	displayConstraintNames();
	qListWidgetPicking->setCurrentRow(pickingNo);
	// m_layerSpectrumDialog->clearHorizon();
	// m_multiSeedSliceRep->addPoint(seed.y, seed.x);
	m_multiSeedSliceRep->addPoint(seed);
	*/
}


std::vector<float*> RgtVolumicDialog::getConstaintHorizons()
{
	std::vector<float*> horizons;
	/*
	int Nh = constraintPoints.size();
	horizons.resize(Nh);
	for (int i=0; i<Nh; i++)
	{
		addPoints(constraintPoints[i]);
		horizons[i] = m_layerSpectrumDialog->getHorizonFromSeed();
	}
	*/
	return horizons;
}

void RgtVolumicDialog::rgtRun()
{
	/*
//	trt_debugClick();
	std::vector<float*> constraintHorizons = getConstaintHorizons();

	int rgtMethode = RgtProcessing::PROCESSING_METHODE::METHODE_VOLUMIC;
	int stack_cpu_gpu = 1;
	int tab_gpu[] = { 0 };
	int tab_gpu_nbre = 1;
	double dip_threshold = 5.0;
	int nbthreads = 1;
	int size[] = { 750, 200, 400 };
	int decimation_factor = 1;
	int nbiter = 20;
	char *dipx_filename0 = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/DATA/SEISMIC/seismic3d.HR_NEAR_cut_dipxy.xt";
	char *dipz_filename0 = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/DATA/SEISMIC/seismic3d.HR_NEAR_cut_dipxz.xt";
	char *seismic_filename0 = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/DATA/SEISMIC/seismic3d.HR_NEAR_cut.xt";
	int bool_polarity = 1;
	int stack_format = SURFACE_STACK_FORMAT_SHORT;
	double sigma_stack = 0.0;
	char *rgt_filename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/DATA/SEISMIC/seismic3d.HR_NEAR_cut__rgtc.xt";
	bool bool_snapping = false;
	double rgt_compresserror = 0.001;
	int seed_threshold = 10000;
	float seismic_start_sample = 0.0f;
	float seismic_step_sample = 1.0f;
	char *constraintsFilename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/DATA/SEISMIC/seismic3d.HR_NEAR_cut_cnx_l256.xt";
	int bool_mask2d = 1;
	FILEIO2::getSeismicStartAndStepSample(seismic_filename0, &seismic_start_sample, &seismic_step_sample);
	FILEIO2 *pf = new FILEIO2();
	pf->createNew(seismic_filename0, rgt_filename, size[1], size[0], size[2], 2);
	delete pf;

	RgtProcessing *pRgt = new RgtProcessing();
	pRgt->setMethode(rgtMethode);
	pRgt->setGPU(stack_cpu_gpu);
	pRgt->setGpuList(tab_gpu, tab_gpu_nbre);
	pRgt->setDipThreshold(dip_threshold);
	pRgt->setNbThreads(nbthreads);
	pRgt->setNativeSize(size);
	pRgt->setDecimationFactor(decimation_factor);
	pRgt->setSurfaceStackNbIter(nbiter);
	pRgt->setDipxyFilename(dipx_filename0);
	pRgt->setDipxzFilename(dipz_filename0);
	pRgt->setSeismicFilename(seismic_filename0);
	pRgt->setSurfaceStackBoolPolarity(bool_polarity);
	pRgt->setSurfaceStackBoolmask2D(bool_mask2d);
	pRgt->setSurfaceStackStackFormat(stack_format);
	pRgt->setSigmaStack(sigma_stack);
	pRgt->setRgtFilename(rgt_filename);

	pRgt->setSnapping(bool_snapping);
	pRgt->setCwtCompressionError(rgt_compresserror);
	pRgt->setSeedThreshold(seed_threshold);
	pRgt->setSeismicStartSample(seismic_start_sample);
	pRgt->setSeismicStepSample(seismic_step_sample);
	pRgt->setPatchConstraintsFilename(constraintsFilename);
	pRgt->setConstraintsSeed(constraintPoints);
	pRgt->setRgtConstraintNbIter(200);
	pRgt->setRgtConstraintEpsilon(0.05);
	pRgt->setConstraintHorizon(constraintHorizons);


	//  GLOBAL_RUN_TYPE = 1;
	pRgt->run();
	delete pRgt;
	*/
}






// ==== D E B U G
void RgtVolumicDialog::trt_debugClick()
{
	// float *h = m_layerSpectrumDialog->getHorizonFromSeed();

	// std::vector<float*> h = getConstaintHorizons();

	// MultiSeedHorizon *tmp = m_multiSeedSliceRep->getData();

	// float *h = m_layerSpectrumDialog->getHorizonFromSeed();
	// h = h;

	/*
	Seismic3DAbstractDataset *m_datasetS = m_layerSpectrumDialog->getDataSetS();
	QString horizonBaseName = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/ImportExport/IJK/HR_NEAR_cut/HORIZON_GRIDS/a.raw";
	QString filenameStd = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/ImportExport/IJK/HR_NEAR_cut/HORIZON_GRIDS/a.raw";

	IJKHorizon* horizon = new IJKHorizon(horizonBaseName, filenameStd, QString::fromStdString(dynamic_cast<Seismic3DDataset*>(m_datasetS)->path()),
					m_datasetS->workingSetManager());
	// m_datasetS->workingSetManager()->addIJKHorizon(horizon);

	int width = horizon->getIsochrone()->getNumTraces();
	int depth = horizon->getIsochrone()->getNumProfils();

	fprintf(stderr, "%d %d\n", width, depth);
	*/

	/*
	const short* rgtBuf = m_layerSpectrumDialog->getMdata()->getModuleData(0);
	for (int i=0; i<10; i++)
		fprintf(stderr, "%d\n", rgtBuf[i]);
		*/


	/*
	char *srcFilename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/DATA/SEISMIC/seismic3d.HR_NEAR.xt";
	char *dstFilename = "/data/PLI/DIR_PROJET/JD-UMC/DATA/3D/UMC_cut/DATA/SEISMIC/seismic3d.HR_NEAR_cut.xt";

	FILEIO2 *p1 = new FILEIO2();
	p1->openForRead(srcFilename);

	int dimx1 = p1->get_dimx();
	int dimy1 = p1->get_dimy();
	int dimz1 = p1->get_dimz();

	int dimx2 = 200;
	int dimy2 = 750;
	int dimz2 = 400;

	FILEIO2 *p2 = new FILEIO2();
	p2->createNew(srcFilename, dstFilename, dimx2, dimy2, dimz2, p1->get_format());
	delete p2;

	p2 = new FILEIO2();
	p2->openForWrite(dstFilename);

	short *data1 = new short[(long)dimx1*dimy1];
	short *data2 = new short[(long)dimx2*dimy2];

	for (int z=0; z<dimz2; z++)
	{
		p1->inlineRead(z, data1);
		for (int y=0; y<dimy2; y++)
			for (int x=0; x<dimx2; x++)
				data2[y*dimx2+x] = data1[y*dimx1+x];
		p2->inlineWrite(z, data2);
	}
	delete p1;
	delete p2;
	delete data1;
	delete data2;
	*/
}
