#include "exportmultilayerblocdialog.h"

#include <iostream>

#include <QListWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QString>
#include <QPushButton>
#include <QListWidget>
#include <QListWidget>
#include <QDialogButtonBox>
#include <QCheckBox>
#include <QLabel>
#include <QTabWidget>
#include <QLineEdit>
#include <QMessageBox>
#include <QRegularExpression>
#include <QRegularExpressionValidator>

#include "rgblayerslice.h"
#include "LayerSlice.h"
#include "abstractgraphicrep.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "fixedrgblayersfromdatasetandcuberep.h"
#include "sismagedbmanager.h"
#include "seismic3ddataset.h"
#include "layerings.h"
#include "cultural.h"
#include "culturals.h"
#include "isochron.h"
#include "stringselectordialog.h"
#include "GraphicsPointerExt.h"
#include "culturalcategory.h"


ExportMultiLayerBlocDialog::ExportMultiLayerBlocDialog(QString title,
		FixedRGBLayersFromDatasetAndCube* dataset, QWidget *parent) :
		m_refDataset(dataset) {

	this->setWindowTitle(title);

	int firstGeotimeNumber = m_refDataset->getIsoOrigin();
	int geotimeStepNumber = m_refDataset->getIsoStep();
	int lastGTNumber = m_refDataset->getIsoOrigin() + (m_refDataset->numLayers()-1) * m_refDataset->getIsoStep();

	QVBoxLayout* pMainLayout = new QVBoxLayout();
	this->setLayout(pMainLayout);

	long currentNumber = firstGeotimeNumber + dataset->currentImageIndex()
			* geotimeStepNumber;

	int rgtMinValue = std::min(firstGeotimeNumber, lastGTNumber);
	int rgtMaxValue = std::max(firstGeotimeNumber, lastGTNumber);

	QHBoxLayout *firstGTL = new QHBoxLayout();
	m_firstGTLE = new QLineEdit();
	m_firstGTLE->setValidator(new QIntValidator(rgtMinValue, rgtMaxValue, this));
	m_firstGTLE->setEnabled(true);
	m_firstGTLE->setText(QString::number(currentNumber));
	firstGTL->addWidget(new QLabel("First Geological Time"));
	firstGTL->addWidget(m_firstGTLE);
	pMainLayout->addLayout(firstGTL);

	QHBoxLayout *stepGTL = new QHBoxLayout();
	m_stepGTLE = new QLineEdit();
	m_stepGTLE->setValidator(new QIntValidator(std::abs(geotimeStepNumber), 100000,this));
	m_stepGTLE->setEnabled(true);
	m_stepGTLE->setText(QString::number(std::abs(m_refDataset->getIsoStep())));
	stepGTL->addWidget(new QLabel("step Geological Time"));
	stepGTL->addWidget(m_stepGTLE);
	pMainLayout->addLayout(stepGTL);

	QHBoxLayout *lastGTL = new QHBoxLayout();
	m_lastGTLE = new QLineEdit();
	m_lastGTLE->setValidator(new QIntValidator(rgtMinValue, rgtMaxValue, this));
	m_lastGTLE->setEnabled(true);
	m_lastGTLE->setText(QString::number(currentNumber) );
	lastGTL->addWidget(new QLabel("Last Geological Time"));
	lastGTL->addWidget(m_lastGTLE);
	pMainLayout->addLayout(lastGTL);

	QHBoxLayout *nameL = new QHBoxLayout();
	nameL->addWidget(new QLabel("Generic Name"));
	QLineEdit *nv_LE = new QLineEdit();
//	nv_LE->setEnabled(false);
//	nv_LE->setText("Nv_");
	nameL->addWidget(new QLabel("Nv_"));
	m_nameLE = new QLineEdit();
	m_nameLE->setEnabled(true);
	QString refDatasetName = QString::fromStdString(
			SismageDBManager::fixCulturalName(m_refDataset->name().toStdString()));
	m_nameLE->setText(refDatasetName);
	QRegularExpression regExp(QString::fromStdString(SismageDBManager::getCulturalRegex()));
	m_nameLE->setValidator(new QRegularExpressionValidator(regExp, this));

	nameL->addWidget(m_nameLE);
	pMainLayout->addLayout(nameL);

	std::string survey3DPath = m_refDataset->surveyPath().toStdString();
	std::pair<double, double> steps = Isochron::getStepFact(survey3DPath, m_refDataset->cubeSeismicAddon());
	if (steps.first>1 || steps.second>1) {
		QHBoxLayout *interpolationL = new QHBoxLayout();
		interpolationL->addWidget(new QLabel("Use Interpolation"));
		m_interpolationCheckBox = new QCheckBox;
		m_interpolationCheckBox->setCheckState(Qt::Checked);
		interpolationL->addWidget(m_interpolationCheckBox);

		pMainLayout->addLayout(interpolationL);
	} else {
		m_interpolationCheckBox = nullptr;
	}

	QDialogButtonBox *buttonBox = new QDialogButtonBox(
			QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	pMainLayout->addWidget(buttonBox);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &ExportMultiLayerBlocDialog::accepted);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

	this->adjustSize();
}

ExportMultiLayerBlocDialog::~ExportMultiLayerBlocDialog() {}

void ExportMultiLayerBlocDialog::accepted() {

	int firstGeotimeNumber = m_refDataset->getIsoOrigin();
	int realRGTStep = m_refDataset->getIsoStep();
	int geotimeStepNumber = std::abs(m_refDataset->getIsoStep());
	int lastGTNumber = m_refDataset->getIsoOrigin() + (m_refDataset->numLayers()-1) * m_refDataset->getIsoStep();

	int rgtMinValue = std::min(firstGeotimeNumber, lastGTNumber);
	int rgtMaxValue = std::max(firstGeotimeNumber, lastGTNumber);

	rgtMinValue = locale().toInt(m_firstGTLE->text());
	geotimeStepNumber = locale().toInt(m_stepGTLE->text());
	rgtMaxValue = locale().toInt(m_lastGTLE->text());

	int firstGeotimeIndex = (rgtMinValue - m_refDataset->getIsoOrigin()) / m_refDataset->getIsoStep();
	int geotimeStepIndex = geotimeStepNumber / m_refDataset->getIsoStep();
	int lastGTIndex = (rgtMaxValue - m_refDataset->getIsoOrigin()) / m_refDataset->getIsoStep();
	int nbCompute = std::abs(lastGTIndex - firstGeotimeIndex) / std::abs(geotimeStepIndex) + 1;
	QString message = "You will create " + QString::number(nbCompute) +
			" Cultural(s) and (Horizon(s). Continue?";
	QMessageBox messageBox;
	messageBox.setText("Export To Sismage");
	messageBox.setInformativeText(message);
	messageBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
	int answer = messageBox.exec();
//	int answer = QMessageBox::question(parent, "Export To Sismage",
//			message.toStdString(), QMessageBox::Yes | QMessageBox::No);
	if (answer != QMessageBox::Yes){
		return;
	}

	QString s( m_nameLE->text());
	QString c = "Nv_";
	if (!s.isNull()&& !s.isEmpty())
		c += s;
	c.replace(" ", "");
	std::string genericName( c.toStdString());

	// Warning it is supposed to be related to sismage Dataset
	std::string survey3DPath = m_refDataset->surveyPath().toStdString();
	// ---- Export to Cultural ----------------------

	std::string culturalsPath =
					SismageDBManager::surveyPath2CulturalPath(survey3DPath);

	if( culturalsPath.empty() ) return;

	int dimH = m_refDataset->width();
	int dimW = m_refDataset->depth();

	bool interpolate = false;
	if (m_interpolationCheckBox) {
		interpolate = m_interpolationCheckBox->checkState()==Qt::Checked;
	}


	// Cultural Category NextVision
	CulturalCategory  nvCulturalCategory(culturalsPath, "NextVision");

	int minimumValue = 0;
	if (m_refDataset->isMinimumValueActive()) {
		minimumValue = std::floor(m_refDataset->minimumValue() * 255);
	}
	for (int indexGT = firstGeotimeIndex; (geotimeStepIndex<0)?(indexGT >= lastGTIndex):(indexGT <= lastGTIndex); indexGT+=geotimeStepIndex) {

		CUDAImagePaletteHolder *red = new CUDAImagePaletteHolder(
				m_refDataset->getNbTraces(), m_refDataset->getNbProfiles(),
						ImageFormats::QSampleType::INT16/*,
						m_data->ijToInlineXlineTransfoForXline(), parent*/);
		CUDAImagePaletteHolder *green = new CUDAImagePaletteHolder(
				m_refDataset->getNbTraces(), m_refDataset->getNbProfiles(),
						ImageFormats::QSampleType::INT16);
		CUDAImagePaletteHolder *blue = new CUDAImagePaletteHolder(
				m_refDataset->getNbTraces(), m_refDataset->getNbProfiles(),
						ImageFormats::QSampleType::INT16);

		CUDAImagePaletteHolder *iso = new CUDAImagePaletteHolder(
				m_refDataset->getNbTraces(), m_refDataset->getNbProfiles(),
						ImageFormats::QSampleType::INT16);

		m_refDataset->getImageForIndex(indexGT, red, green, blue, iso);

		char buf[12];
		sprintf(buf, "_%05d", m_refDataset->isoOrigin()+indexGT*m_refDataset->getIsoStep());
		std::string propName = buf;
		std::string sliceName = genericName + propName;

		Cultural cultural(culturalsPath, sliceName, nvCulturalCategory);
		cultural.saveInto( red, green, blue, minimumValue, m_refDataset->ijToXYTransfo(), true);

		Isochron isochron(sliceName, survey3DPath);
		isochron.saveInto( iso, m_refDataset->cubeSeismicAddon(), interpolate );

		delete red;
		delete green;
		delete blue;
		delete iso;
	}
	accept();
}
