
#ifndef __SPECTRUMPROCESSWIDGET__
#define __SPECTRUMPROCESSWIDGET__


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


#include <vector>
#include <math.h>

#include <ProjectManagerWidget.h>
#include <fileSelectWidget.h>
#include <rgtToRgb16bitsWidget.h>
#include <rgtToSeismicMeanWidget.h>
#include <rgtToGccWidget.h>
#include <rawToAviWidget.h>
#include <aviViewWidget.h>
#include <rgb16ToRgb8Widget.h>
#include <GeotimeSystemInfo.h>
// #include <GeotimeConfiguratorWidget.h>

class AttributToXtWidget;

class SpectrumProcessWidget : public QWidget{
	Q_OBJECT
public:
	enum DATA_OUT_FORMAT { CUBE3D, SINGLE_DATA };
		SpectrumProcessWidget();
		virtual ~SpectrumProcessWidget();
		QString getSeismicName();
		QString getSeismicPath();
		GeotimeSystemInfo *m_systemInfo = nullptr;
		int getDataOutFormat();

private:
		ProjectManagerWidget *m_selectorWidget;
		FileSelectWidget *m_seismicFileSelectWidget;
		RgtToRgb16bitsWidget *m_rgtToRgb16bitsWidget;
		RgtToSeismicMeanWidget *m_rgtToSeismicMeanWidget;
		RgtToGccWidget *m_rgtToGccWidget;
		AttributToXtWidget *m_attributToXtWidget;
		RawToAviWidget *m_rawToAviWidget = nullptr;
		AviViewWidget *m_aviViewWidget = nullptr;
		Rgb16ToRgb8Widget *m_rgb16ToRgb8Widget = nullptr;
		int dataOutFormat = 1;
};



#endif
