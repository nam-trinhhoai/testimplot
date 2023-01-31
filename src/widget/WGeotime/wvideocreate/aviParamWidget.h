

#ifndef __AVIPARAMWIDGET__
#define __AVIPARAMWIDGET__


#include <QThread>
#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QComboBox>
#include <QGroupBox>
#include <QDialog>
#include <QVariant>

#include <vector>
#include <math.h>


#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QSlider>
#include <QPushButton>

class VideoCreateWidget;

class AviParamWidget : public QWidget{
	Q_OBJECT
public:
	AviParamWidget(VideoCreateWidget *videoParamWidget);
	virtual ~AviParamWidget();

	void setVideoScale(double val);
	double getVideoScale();

	QVariant directionCurrentData();
	int getSectionPosition();
	int getSectionPositionMinimum();
	int getSectionPositionSingleStep();
	int getRgtOrigin();
	int getRgtStep();
	bool getRgtreverse();
	bool getOnlyFirstImage();
	void initSectionSlider();
	int getFirstIso();
	int getLastIso();
	// todo
	double videoScale = 1.0;
	int rgb1TOrigin = 0;
	int rgb1TStep = 25;
	int framePerSecond = 25;
	int firstIso = 0;
	int lastIso = 31975;
	int textSize = 24;
	QColor textColor = Qt::white;
	bool rgb1IsReversed = true;

private:
	VideoCreateWidget *m_videoParamWidget = nullptr;
	// QLineEdit *lineedit_aviPrefix = nullptr;
	QDoubleSpinBox *spinboxvideoScale = nullptr;
	QSpinBox *spinboxRgb1TOrigin = nullptr,
			*spinboxRgb1TStep = nullptr,
			*spinboxFps = nullptr,
			*spinboxFirstIso = nullptr,
			*spinboxLastIso = nullptr,
			*sectionPositionSpinBox = nullptr,
			*textSizeSpinBox = nullptr;

	QCheckBox *checkboxRgb1IsReversed = nullptr;
	QComboBox *directionComboBox = nullptr;
	QCheckBox *m_onlyFirstImage = nullptr;

	QSlider *sectionPositionSlider = nullptr;
	QPushButton *colorHolder = nullptr;


	void setAviTextColor(const QColor& color);
	int firstInit = 0;

private slots:
//	void trt_attributTypeChange();
//	void trt_launch();
//	void trt_rgt_Kill();
//
	void trt_rgb1TOriginChanged(int val);
	void trt_rgb1TStepChanged(int val);
	void trt_rgb1IsReversedChanged(int state);
	void trt_fpsChanged(int val);
	void trt_firstIsoChanged(int val);
	void trt_lastIsoChanged(int val);
	void trt_textSizeChanged(int val);
	void trt_directionChanged(int newComboBoxIndex);
	void trt_sectionIndexChanged(int sectionIndex);
	void trt_changeAviTextColor();




};



#endif




