
#include <QVBoxLayout>
#include <QMessageBox>
#include <QDebug>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QInputDialog>
#include <QTemporaryFile>
#include <QColorDialog>


#include <QFileUtils.h>

#include <math.h>
#include <cmath>
#include <iostream>


#include <Xt.h>
#include <fileSelectorDialog.h>
#include <rgtSpectrumHeader.h>
#include <cuda_rgb2torgb1.h>
#include "sismagedbmanager.h"
#include "smsurvey3D.h"
#include "affine2dtransformation.h"
#include "seismic3ddataset.h"
#include "seismicsurvey.h"
#include "smdataset3D.h"
#include <util.h>
#include "globalconfig.h"
#include <cuda_rgt2rgb.h>
#include <spectrumProcessWidget.h>
#include <checkDataSizeMatch.h>
#include <rgtToAttribut.h>
#include <horizonUtils.h>
#include <rawToAviWidget.h>



RawToAviWidget::RawToAviWidget(ProjectManagerWidget *selectorWidget, QWidget* parent)
{

	m_selectorWidget = selectorWidget;
	QVBoxLayout * mainLayout = new QVBoxLayout(this);

	QHBoxLayout *qhbAttributType = new QHBoxLayout;
	QLabel *label_attributType = new QLabel("data type");
	m_attributType = new QComboBox;
	m_attributType->addItem("spectrum");
	m_attributType->addItem("mean");
	m_attributType->addItem("gcc");
	// m_attributType->addItem("mean");
	m_attributType->setMaximumWidth(200);
	qhbAttributType->addWidget(label_attributType);
	qhbAttributType->addWidget(m_attributType);
	qhbAttributType->setAlignment(Qt::AlignLeft);

	/*
	m_rawToAviRgb1FileSelectWidget = new FileSelectWidget();
	m_rawToAviRgb1FileSelectWidget->setProjectManager(m_selectorWidget);
	m_rawToAviRgb1FileSelectWidget->setLabelText("rgt filename");
	m_rawToAviRgb1FileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::seismic);
	m_rawToAviRgb1FileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Rgt);
	m_rawToAviRgb1FileSelectWidget->setLabelDimensionVisible(false);
	*/

	QLabel *labelAttributDirectory = new QLabel("data");
	m_lineEditAttributDirectory = new QLineEdit;
	QPushButton *m_pushButtonAttributDirectory = new QPushButton("...");

	QHBoxLayout *hbAttributDirectory = new QHBoxLayout;
	hbAttributDirectory->addWidget(labelAttributDirectory);
	hbAttributDirectory->addWidget(m_lineEditAttributDirectory);
	hbAttributDirectory->addWidget(m_pushButtonAttributDirectory);

	m_rawToAviRgb2FileSelectWidget = new FileSelectWidget();
	m_rawToAviRgb2FileSelectWidget->setProjectManager(m_selectorWidget);
	m_rawToAviRgb2FileSelectWidget->setLabelText("rgb2 filename");
	m_rawToAviRgb2FileSelectWidget->setFileType(FileSelectWidget::FILE_TYPE::rgtCubeToAttribut);
	m_rawToAviRgb2FileSelectWidget->setFileSortType(FileSelectWidget::FILE_SORT_TYPE::Raw);
	m_rawToAviRgb2FileSelectWidget->setLabelDimensionVisible(false);


	QHBoxLayout *layout42 = new QHBoxLayout;
	QLabel *labelaviPrefix = new QLabel("video prefix:");
	lineedit_aviPrefix = new QLineEdit("video");
	layout42->addWidget(labelaviPrefix);
	layout42->addWidget(lineedit_aviPrefix);

	QHBoxLayout* layout43 = new QHBoxLayout;
	QLabel* labelvideoScale = new QLabel("Scale factor :");
	spinboxvideoScale = new QDoubleSpinBox;
	spinboxvideoScale->setMinimum(std::numeric_limits<double>::min());
	spinboxvideoScale->setMaximum(std::numeric_limits<double>::max());
	spinboxvideoScale->setValue(videoScale);

	layout43->addWidget(labelvideoScale, 0);
	layout43->addWidget(spinboxvideoScale, 1);

	QHBoxLayout* layout44 = new QHBoxLayout();
	QLabel* labelTOrigin = new QLabel("T origin :");
	spinboxRgb1TOrigin = new QSpinBox;
	spinboxRgb1TOrigin->setMinimum(0);
	spinboxRgb1TOrigin->setMaximum(std::numeric_limits<int>::max());
	spinboxRgb1TOrigin->setValue(rgb1TOrigin);

	layout44->addWidget(labelTOrigin, 0);
	layout44->addWidget(spinboxRgb1TOrigin, 1);

	QHBoxLayout* layout45 = new QHBoxLayout();
	QLabel* labelTStep = new QLabel("T Step :");
	spinboxRgb1TStep = new QSpinBox;
	spinboxRgb1TStep->setMinimum(1);
	spinboxRgb1TStep->setMaximum(std::numeric_limits<int>::max());
	spinboxRgb1TStep->setValue(rgb1TStep);

	layout45->addWidget(labelTStep, 0);
	layout45->addWidget(spinboxRgb1TStep, 1);

	QHBoxLayout* layout46 = new QHBoxLayout();
	QLabel* labelIsReversed = new QLabel("Is time reversed :");
	checkboxRgb1IsReversed = new QCheckBox();
	checkboxRgb1IsReversed->setCheckState((rgb1IsReversed) ? Qt::Checked : Qt::Unchecked);

	layout46->addWidget(labelIsReversed, 0);
	layout46->addWidget(checkboxRgb1IsReversed, 1);

	QHBoxLayout* layout47 = new QHBoxLayout();
	QLabel* labelFps = new QLabel("FPS :");
	spinboxFps = new QSpinBox;
	spinboxFps->setMinimum(1);
	spinboxFps->setMaximum(std::numeric_limits<int>::max());
	spinboxFps->setValue(framePerSecond);

	layout47->addWidget(labelFps, 0);
	layout47->addWidget(spinboxFps, 1);

	QHBoxLayout* layout48 = new QHBoxLayout();
	QLabel* labelFirstIso = new QLabel("First Iso :");
	spinboxFirstIso = new QSpinBox;
	spinboxFirstIso->setMinimum(std::numeric_limits<int>::min());
	spinboxFirstIso->setMaximum(std::numeric_limits<int>::max());
	spinboxFirstIso->setValue(firstIso);

	layout48->addWidget(labelFirstIso, 0);
	layout48->addWidget(spinboxFirstIso, 1);

	QHBoxLayout* layout49 = new QHBoxLayout();
	QLabel* labelLastIso = new QLabel("Last Iso :");
	spinboxLastIso = new QSpinBox;
	spinboxLastIso->setMinimum(std::numeric_limits<int>::min());
	spinboxLastIso->setMaximum(std::numeric_limits<int>::max());
	spinboxLastIso->setValue(lastIso);

	layout49->addWidget(labelLastIso, 0);
	layout49->addWidget(spinboxLastIso, 1);

	QHBoxLayout* layout410 = new QHBoxLayout();

	directionComboBox = new QComboBox();
	directionComboBox->addItem("Inline", 0);
	directionComboBox->addItem("Xline", 1);
	sectionPositionSpinBox = new QSpinBox();
	sectionPositionSpinBox->setMinimum(0);
	sectionPositionSpinBox->setMaximum(1);
	sectionPositionSpinBox->setValue(0);
	sectionPositionSlider = new QSlider(Qt::Horizontal);
	sectionPositionSlider->setMinimum(0);
	sectionPositionSlider->setMaximum(1);
	sectionPositionSlider->setValue(0);

	layout410->addWidget(directionComboBox);
	layout410->addWidget(sectionPositionSpinBox);
	layout410->addWidget(sectionPositionSlider);

	QHBoxLayout* layout411 = new QHBoxLayout;
	QSpinBox* textSizeSpinBox = new QSpinBox;
	textSizeSpinBox->setMinimum(1);
	textSizeSpinBox->setMaximum(std::numeric_limits<int>::max());
	textSizeSpinBox->setValue(textSize);
	textSizeSpinBox->setPrefix("Text size : ");
	QLabel* colorLabel = new QLabel("Color :");
	colorHolder = new QPushButton();
	setAviTextColor(textColor);

	layout411->addWidget(textSizeSpinBox);
	layout411->addWidget(colorLabel);
	layout411->addWidget(colorHolder);

	m_progressBar = new QProgressBar();
	m_progressBar->setMinimum(0);
	m_progressBar->setMaximum(100);
	m_progressBar->setValue(0);
	// m_progressBar->setStyleSheet("QProgressBar::chunk{background-color:rgb(200,0,0)}");
	m_progressBar->setTextVisible(true);
	m_progressBar->setFormat("");



	processwatcher_rawtoaviprocess = new ProcessWatcherWidget;

	QPushButton *pushbutton_rawtoaviDisplay = new QPushButton("display");
	QPushButton *pushbutton_rawtoaviRun = new QPushButton("start");


	mainLayout->addLayout(qhbAttributType);
	mainLayout->addLayout(hbAttributDirectory);
	// mainLayout->addWidget(m_rawToAviRgb1FileSelectWidget);
	// mainLayout->addWidget(m_rawToAviRgb2FileSelectWidget);

	mainLayout->addLayout(layout42);
	mainLayout->addLayout(layout43);
	mainLayout->addLayout(layout44);
	mainLayout->addLayout(layout45);
	mainLayout->addLayout(layout46);
	mainLayout->addLayout(layout47);
	mainLayout->addLayout(layout48);
	mainLayout->addLayout(layout49);
	mainLayout->addLayout(layout410);
	mainLayout->addLayout(layout411);
	mainLayout->addWidget(m_progressBar);
	mainLayout->addWidget(processwatcher_rawtoaviprocess);
	mainLayout->addWidget(pushbutton_rawtoaviDisplay);

	mainLayout->addWidget(pushbutton_rawtoaviRun);

	connect(pushbutton_rawtoaviRun, SIGNAL(clicked()), this, SLOT(trt_rawToAviRun()));
	connect(pushbutton_rawtoaviDisplay, SIGNAL(clicked()), this, SLOT(trt_rawToAviDisplay()));

	connect(spinboxvideoScale, SIGNAL(valueChanged(double)), this, SLOT(trt_videoScaleChanged(double)));
	connect(spinboxRgb1TOrigin, SIGNAL(valueChanged(int)), this, SLOT(trt_rgb1TOriginChanged(int)));
	connect(spinboxRgb1TStep, SIGNAL(valueChanged(int)), this, SLOT(trt_rgb1TStepChanged(int)));
	connect(checkboxRgb1IsReversed, SIGNAL(stateChanged(int)), this, SLOT(trt_rgb1IsReversedChanged(int)));
	connect(spinboxFps, SIGNAL(valueChanged(int)), this, SLOT(trt_fpsChanged(int)));
	connect(spinboxFirstIso, SIGNAL(valueChanged(int)), this, SLOT(trt_firstIsoChanged(int)));
	connect(spinboxLastIso, SIGNAL(valueChanged(int)), this, SLOT(trt_lastIsoChanged(int)));
	connect(directionComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(trt_directionChanged(int)));
	connect(sectionPositionSlider, SIGNAL(valueChanged(int)), this, SLOT(trt_sectionIndexChanged(int)));
	connect(sectionPositionSpinBox, SIGNAL(valueChanged(int)), this, SLOT(trt_sectionIndexChanged(int)));
	connect(textSizeSpinBox, SIGNAL(valueChanged(int)), this, SLOT(trt_textSizeChanged(int)));
	connect(colorHolder, SIGNAL(clicked()), this, SLOT(trt_changeAviTextColor()));
	connect(processwatcher_rawtoaviprocess, &ProcessWatcherWidget::processEnded, this, &RawToAviWidget::updateRGBD);
	connect(m_pushButtonAttributDirectory, SIGNAL(clicked()), this, SLOT(trt_attributDirectoryOpen()));

	timer = new QTimer();
	timer->start(1000);
    timer->connect(timer, SIGNAL(timeout()), this, SLOT(showTime()));
}


RawToAviWidget::~RawToAviWidget()
{

}


void RawToAviWidget::setSpectrumProcessWidget(SpectrumProcessWidget *spectrumProcessWidget)
{
	m_spectrumProcessWidget = spectrumProcessWidget;
}

void RawToAviWidget::setAviTextColor(const QColor& color) {
	textColor = color;
	colorHolder->setStyleSheet(QString("QPushButton{ color: %1; background-color: %1; }").arg(color.name()));
}

void RawToAviWidget::trt_attributDirectoryOpen()
{
	std::vector<QString> names = m_selectorWidget->getHorizonIsoValueListName();
	std::vector<QString> path = m_selectorWidget->getHorizonIsoValueListPath();

	FileSelectorDialog dialog(&names, "Select file name");
	int code = dialog.exec();
	if (code==QDialog::Accepted)
	{
		int selectedIdx = dialog.getSelectedIndex();
		if (selectedIdx>=0 && selectedIdx<names.size())
		{
			m_lineEditAttributDirectory->setText(names[selectedIdx]);
			m_attributDirectory = path[selectedIdx];
		}
	}
}


void RawToAviWidget::trt_videoScaleChanged(double val) {
	videoScale = val;
}

void RawToAviWidget::trt_directionChanged(int newComboBoxIndex) {
	if (m_spectrumProcessWidget->getSeismicPath().compare("") == 0) {
		return;
	}

	QVariant dirVariant = directionComboBox->itemData(newComboBoxIndex, Qt::UserRole);
	bool ok;
	int direction = dirVariant.toInt(&ok);
	if (!ok) {
		return;
	}

	int size[3];
	double steps[3];
	double origins[3];
	bool isValid = getPropertiesFromDatasetPath(m_spectrumProcessWidget->getSeismicPath(), size, steps, origins);
	if (!isValid) {
		return;
	}

	// inline
	int axisIndex = 2;
	if (direction == 1) {
		// xline
		axisIndex = 0;
	}
	int axisStart = origins[axisIndex];
	int axisStep = steps[axisIndex];
	int axisEnd = axisStart + (size[axisIndex]-1) * axisStep;

	QSignalBlocker bSlider(sectionPositionSlider);
	QSignalBlocker bSpinBox(sectionPositionSpinBox);
	sectionPositionSlider->setMinimum(axisStart);
	sectionPositionSlider->setMaximum(axisEnd);
	sectionPositionSlider->setSingleStep(axisStep);
	sectionPositionSlider->setValue(axisStart);

	sectionPositionSpinBox->setMinimum(axisStart);
	sectionPositionSpinBox->setMaximum(axisEnd);
	sectionPositionSpinBox->setSingleStep(axisStep);
	sectionPositionSpinBox->setValue(axisStart);
}

void RawToAviWidget::trt_sectionIndexChanged(int sectionIndex) {
	trt_directionChanged(directionComboBox->currentIndex());
	QSignalBlocker bSlider(sectionPositionSlider);
	QSignalBlocker bSpinBox(sectionPositionSpinBox);

	sectionPositionSlider->setValue(sectionIndex);
	sectionPositionSpinBox->setValue(sectionIndex);
}

void RawToAviWidget::trt_changeAviTextColor() {
	QColorDialog dialog(textColor);
	int errCode = dialog.exec();
	if (errCode==QDialog::Accepted) {
		setAviTextColor(dialog.selectedColor());
	}
}

void RawToAviWidget::computeoptimalscale_rawToAvi() {
	if (m_spectrumProcessWidget->getSeismicPath().isNull() || m_spectrumProcessWidget->getSeismicPath().isEmpty() || m_attributDirectory.isNull() ||
			m_attributDirectory.isEmpty()) {
		return;
	}

	aviFilenameUpdate();
	int size[3];
	double steps[3];
	bool isValid = getPropertiesFromDatasetPath(m_spectrumProcessWidget->getSeismicPath(), size, steps);
	if (!isValid) {
		return;
	}
	QString logoPath = GlobalConfig::getConfig().logoPath();
	std::tuple<double, bool, double> angleAndSwapV = getAngleFromFilename(m_spectrumProcessWidget->getSeismicPath());
	double ratioWH = std::get<2>(angleAndSwapV);// * ((double) size[0]) / size[2];
	double scaleX=1.0, scaleY=1.0;
	if (ratioWH>1) {
		scaleX = ratioWH;
	} else if (ratioWH!=0.0) {
		scaleY = 1.0 / ratioWH;
	}

	double wOri = size[0] * scaleX;
	double hOri = size[2] * scaleY;
	QSizeF wAndH = newSizeFromSizeAndAngle(QSizeF(wOri, hOri), std::get<0>(angleAndSwapV));
	double w = wAndH.width();
	double h = wAndH.height();

//	double wRatio = w / wOri;
//	double hRatio = h / hOri;

	// total video width has 1/4 for section and 3/4 for map
	// remove from limit the padding 20 top, bottom, left, right
	// padding 10 for the separation
	int borderPadding = 20;
	int centerPadding = 10;
	double ratioWLimit = std::ceil(w * 4.0/3.0) / (7680.0 - borderPadding*2 - centerPadding); // test map width taking into account added section to the limit check
	double ratioHLimit = h / (4320.0 - borderPadding*2);

	double worstRatio = std::max(ratioWLimit, ratioHLimit);

	if (worstRatio>0.75) {
		spinboxvideoScale->setValue(0.75 / worstRatio);
	} else {
		spinboxvideoScale->setValue(1);
	}
}

std::size_t RawToAviWidget::getFileSize(const char* filePath) {
	FILE* f = fopen(filePath, "r");
	std::size_t size = 0;
	if (f!=NULL) {
		fseek(f, 0L, SEEK_END);
		size = ftell(f);
		fclose(f);
	}

	return size;
}

void RawToAviWidget::trt_rgb1TOriginChanged(int val) {
	rgb1TOrigin = val;
}

void RawToAviWidget::trt_rgb1TStepChanged(int val) {
	rgb1TStep = val;
}

void RawToAviWidget::trt_rgb1IsReversedChanged(int state) {
	rgb1IsReversed = state == Qt::Checked;
}

void RawToAviWidget::trt_fpsChanged(int val) {
	framePerSecond = val;
}

void RawToAviWidget::trt_firstIsoChanged(int val) {
	firstIso = val;
}

void RawToAviWidget::trt_lastIsoChanged(int val) {
	lastIso = val;
}

void RawToAviWidget::trt_textSizeChanged(int val) {
	textSize = val;
}

QString RawToAviWidget::formatColorForFFMPEG(const QColor& color) {
	QString qtStyledColor = color.name();
	// remove "#" and add "0x"
	QString ffmpegColor = qtStyledColor;
	ffmpegColor.remove(0, 1);
	ffmpegColor.prepend("0x");
	return ffmpegColor;
}


void RawToAviWidget::updateRGBD() {
	// m_selectorWidget->global_rgb_database_update();
	m_selectorWidget->RgbRawUpdateNames();
}


void RawToAviWidget::aviFilenameUpdate()
{

	if ( m_spectrumProcessWidget->getSeismicName().compare("") == 0 || m_attributDirectory.compare("") == 0 ) return;
	// QString path = filenameToPath(m_rawToAviRgb1FileSelectWidget->getPath());
	int size[3];

	QString tinyName;
	QStringList list = m_lineEditAttributDirectory->text().split(" ");// because there is no display name...
	if (list.size()>0) {
		tinyName = list[0];
	}

	// aviFullName = path + "/" + tinyName + "_" + lineedit_aviPrefix->text() + ".avi";
	// aviFullName = m_rawToAviRgb1FileSelectWidget->getPath() + "/../" + tinyName + "_" + lineedit_aviPrefix->text() + ".avi";

	QString seismicName = m_spectrumProcessWidget->getSeismicName();
	aviFullName = m_selectorWidget->getIJKPath() + "/" + seismicName + "/cubeRgt2RGB/" + tinyName + "_" + lineedit_aviPrefix->text() + ".avi";

	mkdirPathIfNotExist(m_selectorWidget->getIJKPath());
	mkdirPathIfNotExist(m_selectorWidget->getIJKPath()+ "/" + seismicName);
	mkdirPathIfNotExist(m_selectorWidget->getIJKPath()+ "/" + seismicName + + "/cubeRgt2RGB/");

	qDebug() << aviFullName;
	QFileInfo info(aviFullName);
	aviTinyName = info.fileName();
}


bool RawToAviWidget::getPropertiesFromDatasetPath(QString filename, int* size, double* steps, double* origins)
{
	if ( filename.compare("") == 0 ) { for (int i=0; i<3;i++) size[i] = 0; return false; }
	inri::Xt xt(filename.toStdString().c_str());
	if (!xt.is_valid()) {
		std::cerr << "xt cube is not valid (" << filename.toStdString() << ")" << std::endl;
		return false;
	}
    size[0] = xt.nRecords();
    size[1] = xt.nSamples();
    size[2] = xt.nSlices();
    steps[0] = xt.stepRecords();
    steps[1] = xt.stepSamples();
    steps[2] = xt.stepSlices();
    if (origins!=nullptr) {
        origins[0] = xt.startRecord();
        origins[1] = xt.startSamples();
        origins[2] = xt.startSlice();
    }

    if ( (size[0] == 0 && size[1] == 0 && size[2] == 0) ||
            steps[0]==0 || steps[1]==0 || steps[2]==0) return false;
    return true;
}


std::tuple<double, bool, double> RawToAviWidget::getAngleFromFilename(QString seismicFullName)
{
	bool topoExist;
	double angle = 0.0;
	bool swapV = true; // swap by default
	std::string surveyPath = SismageDBManager::survey3DPathFromDatasetPath(seismicFullName.toStdString());
	SmSurvey3D survey(surveyPath);

	SeismicSurvey baseSurvey(nullptr, "", survey.inlineDim(), survey.xlineDim(), QString::fromStdString(surveyPath));
	baseSurvey.setInlineXlineToXYTransfo(survey.inlineXlineToXYTransfo());
	baseSurvey.setIJToXYTransfo(survey.ijToXYTransfo());

	Seismic3DDataset seismic(&baseSurvey, "", nullptr, Seismic3DDataset::CUBE_TYPE::Seismic);
	seismic.loadFromXt(seismicFullName.toStdString());
	SmDataset3D d3d(seismicFullName.toStdString());
	seismic.setIJToInlineXlineTransfo(d3d.inlineXlineTransfo());
	seismic.setIJToInlineXlineTransfoForInline(
			d3d.inlineXlineTransfoForInline());
	seismic.setIJToInlineXlineTransfoForXline(
			d3d.inlineXlineTransfoForXline());
	seismic.setSampleTransformation(d3d.sampleTransfo());

	const Affine2DTransformation& transform = *(seismic.ijToXYTransfo());
	double oriX, oriY, endX, endY, vectX, vectY;
	transform.imageToWorld(0, 0, oriX, oriY);
	transform.imageToWorld(0, 1, endX, endY);

	vectX = endX - oriX;
	vectY = endY - oriY;
	double refVectX = 1;
	double refVectY = 0;
	double normVect = std::sqrt(std::pow(vectX, 2) + std::pow(vectY, 2));
	double normRefVect = std::sqrt(std::pow(refVectX, 2) + std::pow(refVectY, 2));

	angle = std::atan(vectY / vectX) - M_PI / 2;
	if (vectX<0) {
		angle += M_PI;
	}
	// std::acos((vectX * refVectX + vectY * refVectY) / ( normVect * normRefVect));

	double orthoX, orthoY;
	transform.imageToWorld(1, 0, orthoX, orthoY);
	double vectOrthoX = orthoX - oriX;
	double vectOrthoY = orthoY - oriY;

	double crossProduct = vectY * vectOrthoX - vectX * vectOrthoY;
	if (crossProduct>=0) {
		swapV = true;
	} else {
		swapV = false;
		angle = angle + M_PI;
	}

	double ratioWH = std::sqrt(std::pow(vectOrthoX, 2) + std::pow(vectOrthoY, 2)) / std::sqrt(std::pow(vectX, 2) + std::pow(vectY, 2));

	std::tuple<double, bool, double> angleAndSwapV(angle, swapV, ratioWH);
	return angleAndSwapV;
}


QSizeF RawToAviWidget::newSizeFromSizeAndAngle(QSizeF oriSize, double angle) {
	QTransform transfo;
	transfo.rotate(angle * 180 / M_PI);
	QPointF p1(0, 0), p2(0, oriSize.height()), p3(oriSize.width(), 0), p4(oriSize.width(), oriSize.height());
	QList<QPointF> newPts;
	newPts << transfo.map(p1);
	newPts << transfo.map(p2);
	newPts << transfo.map(p3);
	newPts << transfo.map(p4);

	double xmin = newPts[0].x(), xmax=newPts[0].x(), ymin=newPts[0].y(), ymax=newPts[0].y();
	for (QPointF pt : newPts) {
		if (pt.x()<xmin) {
			xmin = pt.x();
		}
		if (pt.x()>xmax) {
			xmax = pt.x();
		}
		if (pt.y()<ymin) {
			ymin = pt.y();
		}
		if (pt.y()>ymax) {
			ymax = pt.y();
		}
	}
	return QSizeF(xmax - xmin, ymax - ymin);
}

QString RawToAviWidget::filenameToPath(QString fullName)
{
	int lastPoint = fullName.lastIndexOf("/");
	QString path = fullName.left(lastPoint);
	return path;
}

std::pair<QString, QString> RawToAviWidget::getPadCropFFMPEG(double wRatio, double hRatio)
{
	double epsilon = 1.0e-30;
	bool wEqual = std::fabs(wRatio - 1.0) < epsilon;
	bool wPad = wRatio>1 && !wEqual;
	bool wCrop = wRatio<1 && !wEqual;
	bool hEqual = std::fabs(hRatio - 1.0) < epsilon;
	bool hPad = hRatio>1 && !hEqual;
	bool hCrop = hRatio<1 && !hEqual;

	QString padCmd, cropCmd;
	if (wPad && hPad) {
		padCmd = "pad=" + QString::number(wRatio) + "*iw:" + QString::number(hRatio) + "*ih:(ow-iw)/2:(oh-ih)/2";
	} else if (wCrop && hCrop) {
		cropCmd = "crop=" + QString::number(wRatio) + "*iw:" + QString::number(hRatio) + "*ih:(iw-ow)/2:(ih-oh)/2";
	} else {
		if (wPad) {
			padCmd = "pad=" + QString::number(wRatio) + "*iw:ih:(ow-iw)/2:(oh-ih)/2";
		} else if (wCrop) {
			cropCmd = "crop=" + QString::number(wRatio) + "*iw:ih:(iw-ow)/2:(ih-oh)/2";
		}
		if (hPad) {
			padCmd = "pad=iw:" + QString::number(hRatio) + "*ih:(ow-iw)/2:(oh-ih)/2";
		} else if (hCrop) {
			cropCmd = "crop=iw:" + QString::number(hRatio) + "*ih:(iw-ow)/2:(ih-oh)/2";
		}
	}

	return std::pair<QString, QString>(padCmd, cropCmd);
}

QString RawToAviWidget::getFFMPEGTime(double _time) {
	QString sign = "";
	if (_time<0) {
		sign = "-";
	}
	double time = std::fabs(_time);
	int timeInt = std::floor(time);
	double ms = time - timeInt;
	int s = timeInt % 60;
	int minutes = (timeInt / 60) % 60;
	int hours = (timeInt / 3600);

	return sign + formatTimeWithMinCharacters(hours) + ":" + formatTimeWithMinCharacters(minutes) + ":" +
			formatTimeWithMinCharacters(s+ms);
}

QString RawToAviWidget::formatTimeWithMinCharacters(double time, int minCharNumber) {
	int zerosToAdd = minCharNumber;
	double absTime = std::fabs(time);
	double movingTime = absTime;
	while (movingTime>=1) {
		movingTime = movingTime / 10.0f;
		zerosToAdd--;
	}

	if (minCharNumber==zerosToAdd) {
		zerosToAdd = zerosToAdd - 1; // because QString::number will start by a zero
	}

	QStringList zeros;
	for (int i=0; i<zerosToAdd; i++) {
		zeros << "0";
	}
	int precision = 3;
	double decimals = std::floor(absTime) - absTime;
	if (qFuzzyIsNull(decimals)) {
		precision = 0;
	}

	return zeros.join("") + QString::number(time, 'f', precision);
}


void RawToAviWidget::rawToAviRunFfmeg(bool onlyFirstImage)
{
	QString ImportExportPath = m_selectorWidget->getImportExportPath();
	QString IJKPath = m_selectorWidget->getIJKPath();
	QString rgtPath = m_attributDirectory; // IJKPath + "/" + m_lineEditAttributDirectory->text() + "/";

	// if function changed update computeoptimalscale_rawToAvi
	computeoptimalscale_rawToAvi();

	GlobalConfig& config = GlobalConfig::getConfig();
	QDir tempDir = QDir(config.tempDirPath());

	aviFilenameUpdate();
	if (!onlyFirstImage && QFileInfo(aviFullName).exists()) {
		QStringList overWriteOptions;
		QString noStr = "no";
		overWriteOptions << noStr << "yes";
		bool ok;
		QString val = QInputDialog::getItem(this, "Overwrite ?", "Output file already exists, do you want to overwrite it ?",
				overWriteOptions, 0, false, &ok);

		// no -> abort function
		if (!ok || val.compare(noStr)==0) {
			return;
		}
	}

	QString jpgTmpPath;
	if (onlyFirstImage) {
		QTemporaryFile outJpgImageFile;
		outJpgImageFile.setFileTemplate(tempDir.absoluteFilePath("NextVision_display_XXXXXX.jpg"));
		outJpgImageFile.setAutoRemove(false);
		outJpgImageFile.open();
		outJpgImageFile.close();
		jpgTmpPath = outJpgImageFile.fileName();
	}

	int size[3];
	double steps[3];
	bool isValid = getPropertiesFromDatasetPath(m_spectrumProcessWidget->getSeismicPath(), size, steps);
	if (!isValid) {
		return;
	}

	QVariant dirVariant = directionComboBox->currentData(Qt::UserRole);
	bool ok;
	int dirData = dirVariant.toInt(&ok);
	if (!ok) {
		return;
	}
	QString sectionName;
	SliceDirection direction;
	if (dirData==0) {
		direction = SliceDirection::Inline;
		sectionName = "Inline ";
	} else {
		direction = SliceDirection::XLine;
		sectionName = "XLine ";
	}

	int axisValue = sectionPositionSpinBox->value();
	int axisValueIndex = (axisValue - sectionPositionSpinBox->minimum()) / sectionPositionSpinBox->singleStep();

	sectionName += QString::number(axisValue);

	QTemporaryFile outSectionVideoRawFile;
	outSectionVideoRawFile.setFileTemplate(tempDir.absoluteFilePath("NextVision_section_XXXXXX.rgb"));
	// outSectionVideoRawFile.setFileTemplate(QDir("/data/PLI/NKDEEP/jacques/").absoluteFilePath("NextVision_section_XXXXXX.rgb"));
	outSectionVideoRawFile.setAutoRemove(false);
	outSectionVideoRawFile.open();
	outSectionVideoRawFile.close();
	QString inRgb2Path = m_attributDirectory; //rgb2_Avi_FullName;// "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/ImportExport/IJK/HR_NEAR/cubeRgt2RGB/rgb2_spectrum_mzr_from_HR_NEAR_size_1500x700x1280.raw";
	// bool sectionValid = SectionToVideo::run(m_spectrumProcessWidget->getSeismicPath(), axisValueIndex, direction, inRgb2Path, outSectionVideoRawFile.fileName());
	std::vector<std::string> isoPath = getIsoPath(rgtPath);


	qDebug() << inRgb2Path;


	/*
	QTemporaryFile rgb1Filename;
	rgb1Filename.setFileTemplate(tempDir.absoluteFilePath("NextVision_section_XXXXXX.rgb"));
	// outSectionVideoRawFile.setFileTemplate(QDir("/data/PLI/NKDEEP/jacques/").absoluteFilePath("NextVision_section_XXXXXX.rgb"));
	rgb1Filename.setAutoRemove(false);
	rgb1Filename.open();
	rgb1Filename.close();
	*/

	// QString tmpFilename = "/data/PLI/DIR_PROJET/UMC-NK/DATA/3D/UMC_small/ImportExport/IJK/HR_NEAR/cubeRgt2RGB/rgb1__from_rgb2_spectrum_test_as2_from_HR_NEAR_size_1500x700x1280__alpha_1x0__ratio_x0001_size_1500x700x1280_123456.rgb";
	// QString tmpFilename = inRgb2Path + "_rgb1.rgb";

	/*
	QDir tempDirRgb1 = QDir(m_rawToAviRgb1FileSelectWidget->getPath());
	QTemporaryFile rgb1RawFile;
	rgb1RawFile.setFileTemplate(tempDirRgb1.absoluteFilePath("NextVision_rgb1_XXXXXX.rgb"));
	// outSectionVideoRawFile.setFileTemplate(QDir("/data/PLI/NKDEEP/jacques/").absoluteFilePath("NextVision_section_XXXXXX.rgb"));
	rgb1RawFile.setAutoRemove(false);
	rgb1RawFile.open();
	rgb1RawFile.close();
	QString tmpFilename = rgb1RawFile.fileName();
	*/

	QString tmpFilename = getTmpFilename(rgtPath, "rgb1_", ".rgb");

	qDebug() << outSectionVideoRawFile.fileName();
	qDebug() << tmpFilename;

	// cuda_rgb2torgb1ByDirectories(isoPath, size[0], size[2], isoPath.size(), .0001f, 1.0f, (char*)rgb1Filename.fileName().toStdString().c_str(), nullptr);

	QString dataType = "";
	if ( m_attributType->currentText().compare("spectrum") == 0 )
	{
		dataType = SPECTRUM_NAME;
	}
	else if ( m_attributType->currentText().compare("gcc") == 0 )
	{
		dataType = GCC_NAME;
	}
	else if ( m_attributType->currentText().compare("mean") == 0 )
	{
		dataType = MEAN_NAME;
	}

	if ( !pIhm2 ) pIhm2 = new Ihm2();
	valStartStop = 1;

	// todo
	/*
	cuda_rgb2torgb1ByDirectories(isoPath, size[0], size[2], isoPath.size(), .0001f, 1.0f, (char*)tmpFilename.toStdString().c_str(), dataType.toStdString(),
			m_spectrumProcessWidget->getSeismicName().toStdString(), pIhm2);
			*/

	bool sectionValid = SectionToVideo::run2(isoPath, m_spectrumProcessWidget->getSeismicPath(),
			m_spectrumProcessWidget->getSeismicName(),
			axisValueIndex, direction, inRgb2Path, outSectionVideoRawFile.fileName());
	int axisSize = size[0];
	if (direction == SliceDirection::XLine) {
		axisSize = size[2];
	}

	QString logoPath = GlobalConfig::getConfig().logoPath();
	std::tuple<double, bool, double> angleAndSwapV = getAngleFromFilename(m_spectrumProcessWidget->getSeismicPath());
	double ratioWH = std::get<2>(angleAndSwapV);// * ((double) size[0]) / size[2];
	double scaleX=1.0, scaleY=1.0;
	if (ratioWH>1) {
		scaleX = ratioWH;
	} else if (ratioWH!=0.0) {
		scaleY = 1.0 / ratioWH;
	}

	double wOri = size[0] * scaleX;
	double hOri = size[2] * scaleY;
	QSizeF wAndH = newSizeFromSizeAndAngle(QSizeF(wOri, hOri), std::get<0>(angleAndSwapV));
	double w = wAndH.width();
	double h = wAndH.height();

	int sectionHeight = std::floor(h);
	int sectionWidth = std::round(w/3.0);

	double wRatio = w / wOri;
	double hRatio = h / hOri;

	QString inlineAngleName = QString::number(std::get<0>(angleAndSwapV));
	std::pair<QString, QString> padAndCrop = getPadCropFFMPEG(wRatio, hRatio);

	int tOrigin = rgb1TOrigin;
	int tStep = rgb1TStep; // expect positive int
	bool isReversed = rgb1IsReversed;

	// std::size_t sz = isoPath.size(); //getFileSize(m_rawToAviRgb1FileSelectWidget->getPath().toStdString().c_str());
	// int numFrames = sz / (size[0] * size[2] * sizeof(char) * 3);
	int numFrames = isoPath.size();

	int firstIndex, lastIndex, cutOrigin;
	QString tSign;
	if (isReversed) {
		tOrigin = tOrigin + (numFrames-1) * tStep;
		firstIndex = std::min(std::max((tOrigin - firstIso) / tStep, 0), numFrames-1);
		lastIndex = std::min(std::max((tOrigin - lastIso) / tStep, 0), numFrames-1);

		cutOrigin = tOrigin - std::min(firstIndex, lastIndex) * tStep;
		tSign = "-";
	} else {
		firstIndex = std::min(std::max((firstIso - tOrigin) / tStep, 0), numFrames-1);
		lastIndex = std::min(std::max((lastIso - tOrigin) / tStep, 0), numFrames-1);
		tSign = "+";

		cutOrigin = tOrigin + std::min(firstIndex, lastIndex) * tStep;
	}
	double firstTime = ((double) std::min(firstIndex, lastIndex)) / framePerSecond;
	double lastTime = ((double) std::max(firstIndex, lastIndex) + 1) / framePerSecond;
	double duration = lastTime - firstTime;
	QString beginTime = getFFMPEGTime(firstTime);

	QString drawTextExpr = "%{expr_int_format\\:" + QString::number(cutOrigin) + tSign +
			QString::number(tStep) + "*\\n\\:d\\:5}";

	QStringList options;
	options << "-y" << "-ss" << beginTime << "-t" << QString::number(duration);
	options << "-pixel_format" << "rgb24" << "-vcodec" << "rawvideo" << "-video_size";
	options << (QString::number(size[0]) + "x" + QString::number(size[2])) << "-framerate";
	options << QString::number(framePerSecond) << "-i" << tmpFilename; // rgb1Filename.fileName(); // m_rawToAviRgb1FileSelectWidget->getPath();
	options << "-pixel_format" << "rgb24" << "-vcodec" << "rawvideo" << "-video_size";
	options << (QString::number(axisSize) + "x" + QString::number(size[1])) << "-framerate";
	options << QString::number(framePerSecond) << "-i" << outSectionVideoRawFile.fileName();
	if (!onlyFirstImage) {
		options << "-c:v" << "libx264" << "-crf" << "23";
	}
	options << "-filter_complex";
	QString filterOption = "movie=" + logoPath + " [logo]; [logo] scale=100:-1 [logoBis]; [0] ";
	if (!std::get<1>(angleAndSwapV)) {
		filterOption += "vflip,";
	}
	if (!padAndCrop.first.isNull() && !padAndCrop.first.isEmpty()) {
		filterOption += padAndCrop.first + ",";
	}
	filterOption += "scale=iw*" + QString::number(videoScale * scaleX) + ":ih*" + QString::number(videoScale * scaleY) +
			",rotate=" + inlineAngleName;
	if (!padAndCrop.second.isNull() && !padAndCrop.second.isEmpty()) {
		filterOption += "," + padAndCrop.second;
	}
	filterOption += ",scale='iw-mod(iw,2)':'ih-mod(ih,2)'";
	filterOption += ",vflip";
	filterOption += ",drawtext=fontfile=/usr/share/fonts/gnu-free/FreeSans.ttf:text='RGT time \\: " + drawTextExpr + "':fontsize="+QString::number(textSize)+":fontcolor="+formatColorForFFMPEG(textColor)+":x=h/2-th/2:y=h-th-10";
	filterOption += " [tmp]; [tmp][logoBis] overlay=(main_w-overlay_w-10):(main_h-overlay_h-10) [last]; ";
	filterOption += "[1] scale=" + QString::number(std::floor(videoScale*sectionWidth)-50) + ":" + QString::number(std::floor(videoScale*sectionHeight));
	filterOption += ",drawtext=fontfile=/usr/share/fonts/gnu-free/FreeSans.ttf:text='" + sectionName + "':fontsize="+QString::number(textSize)+":fontcolor="+formatColorForFFMPEG(textColor)+":x=h/2-th/2:y=h-th-10 [last1]; ";
	filterOption += "[last] pad=iw+" + QString::number(std::floor(videoScale*sectionWidth)) + ":ih+40:ow-iw-20:20 [last_padded]; [last_padded][last1] overlay=20:20";

	options << filterOption;
	if (onlyFirstImage) {
		options << "-frames:v" << "1" << "-f" << "image2";
		options << jpgTmpPath;
	} else {
		options << aviFullName;
	}

	//fprintf(stderr, "command: %s\n", cmd.toStdString().c_str());
	qDebug() << "ffmpeg command : " << "ffmpeg " << options.join(" ");

	if (processwatcher_rawtoaviprocess->processState()==QProcess::NotRunning && m_processwatcher_connection==nullptr) {
		QString outSectionVideoRawFilePath = outSectionVideoRawFile.fileName();
		QMetaObject::Connection conn = connect(processwatcher_rawtoaviprocess, &ProcessWatcherWidget::processEnded, [this, outSectionVideoRawFilePath, jpgTmpPath]() {
			QFile::remove(outSectionVideoRawFilePath);
			if (m_processwatcher_connection!=nullptr) {
				QObject::disconnect(*m_processwatcher_connection);
				m_processwatcher_connection.reset(nullptr);
			}

			QProcess* process = new QProcess();
			process->setProgram("display");
			process->setArguments(QStringList() << jpgTmpPath);

			connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), [process, jpgTmpPath]() {
				process->deleteLater();
				QFile::remove(jpgTmpPath);
			});
			connect(process, &QProcess::errorOccurred, [process, jpgTmpPath]() {
				process->deleteLater();
				QFile::remove(jpgTmpPath);
			});

			process->start();
		});
		m_processwatcher_connection.reset(new QMetaObject::Connection(conn));
		processwatcher_rawtoaviprocess->launchProcess("ffmpeg", options);
	}
	//system(cmd.toStdString().c_str());
	// system("ls");
	// system("ffmpeg -f rawvideo -c:v rawvideo -pix_fmt rgb24 -s:v 2000x400 -r 10 -i /data/PLI/DIR_PROJET/JD_TEST/DATA/3D/JD/DATA/SEISMIC/seismic__spectrum_rgb2_wsize_64_f1_2_f2_4_f3_6__rgb1__alpha_1.0__ratio_.0001.raw -c:v libx264 /data/PLI/jacques/idle0/output0.mpeg");
	//m_selectorWidget->global_rgb_database_update();
	valStartStop = 0;
	if ( m_selectorWidget ) m_selectorWidget->RgbRawUpdateNames();
}


void RawToAviWidget::trt_rawToAviRun() {
	rawToAviRunFfmeg(false);
	/*
	RawToAviWidgetTHREAD *thread = new RawToAviWidgetTHREAD(this, false);
	thread->start();
	thread->wait();
	fprintf(stderr, "ok\n");
	*/
}

void RawToAviWidget::trt_rawToAviDisplay() {
	rawToAviRunFfmeg(true);
	/*
	RawToAviWidgetTHREAD *thread = new RawToAviWidgetTHREAD(this, true);
	thread->start();
	thread->wait();
	fprintf(stderr, "ok\n");
	*/
}


std::vector<std::string> RawToAviWidget::getIsoPath(QString path)
{
	// todo
	// if ( m_attributDirectoty->getPath().isEmpty() ) return std::vector<std::string>();
	std::vector<std::string> out;

	QDir *dir = new QDir(path);
	dir->setFilter(QDir::Dirs| QDir::NoDotAndDotDot);
	// dir->setSorting(QDir::Name);
	QFileInfoList list = dir->entryInfoList();
	for (int n=0; n<list.size(); n++)
	{
		QString path0 = list.at(n).fileName();
		if ( path0.contains("iso_") )
		{
			QString tmp = path + "/" + path0 + "/";
			out.push_back(tmp.toStdString());
		}
	}
	return out;
}


void RawToAviWidget::showTime()
{
	if ( !pIhm2 ) return;
	if ( valStartStop == 1 && pIhm2->isSlaveMessage() )
	{
		Ihm2Message mess = pIhm2->getSlaveMessage();
		float val_f = 100.0*mess.count/mess.countMax;
		int val = (int)(val_f);
		m_progressBar->setValue(val);
		m_progressBar->setStyleSheet("QProgressBar::chunk{background-color:rgb(0,200,0)}");
		QString text = QString::fromStdString(mess.message) + " " + QString::number(val_f, 'f', 1) + "%";
		m_progressBar->setFormat(text);
	}
	else if ( valStartStop == 0 )
	{
		m_progressBar->setValue(0);
		m_progressBar->setFormat("");
	}
}


RawToAviWidgetTHREAD::RawToAviWidgetTHREAD(RawToAviWidget *p, bool _val)
 {
     this->pp = p;
     this->val = _val;
 }

 void RawToAviWidgetTHREAD::run()
 {
	 fprintf(stderr, "thread start\n");
	 pp->rawToAviRunFfmeg(val);
 }
