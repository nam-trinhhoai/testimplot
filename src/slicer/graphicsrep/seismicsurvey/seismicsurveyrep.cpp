#include "seismicsurveyrep.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismicsurveyproppanel.h"
#include "seismicsurveylayer.h"
#include "slicepositioncontroler.h"
#include "qgllineitem.h"
#include <QMenu>
#include <QAction>
#include <QProgressDialog>
#include <QMessageBox>
#include "workingsetmanager.h"
#include "abstractinnerview.h"
#include "composeseismictorgbdialog.h"
#include "rgbdataset.h"
#include "seismic3ddataset.h"
#include "smdataset3D.h"
#include "stringselectordialog.h"
#include "QFileDialog"
#include "DataUpdatorDialog.h"
#include "seismicinformationaggregator.h"
#include "managerwidget.h"


SeismicSurveyRep::SeismicSurveyRep(SeismicSurvey *survey,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_survey = survey;
	m_name = m_survey->name();
}

SeismicSurveyRep::~SeismicSurveyRep() {

}
IData* SeismicSurveyRep::data() const {
	return m_survey;
}

QWidget* SeismicSurveyRep::propertyPanel() {
	return nullptr;
}
GraphicLayer* SeismicSurveyRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	return nullptr;
}
void SeismicSurveyRep::buildContextMenu(QMenu *menu) {

	QAction *infoAction = new QAction(tr("Information"), this);
	menu->addAction(infoAction);
	connect(infoAction, SIGNAL(triggered()), this, SLOT(openIsoHorizonInformation()));

	QAction *rgbAction = new QAction(tr("RGB Composite"), this);
	menu->addAction(rgbAction);
	connect(rgbAction, SIGNAL(triggered()), this, SLOT(createRgbComposite()));

	QAction *addAction = new QAction(tr("Add seismic"), this);
	menu->addAction(addAction);
	connect(addAction, SIGNAL(triggered()), this, SLOT(AddSeismic()));
}


void SeismicSurveyRep::createRgbComposite() {
	if (m_survey->datasets().size()>0) {
		ComposeSeismicToRgbDialog dialog(m_survey);
		int code = dialog.exec();
		if (code==QDialog::Accepted) {
			RgbDataset* rgbDataset = RgbDataset::createRgbDataset(QString("Rgb on ")+m_survey->name(),
					dialog.red(), dialog.channelRed(), dialog.green(), dialog.channelGreen(),
					dialog.blue(), dialog.channelBlue(), dialog.alpha(), dialog.channelAlpha(),
					m_survey->workingSetManager());
			if (rgbDataset!=nullptr) {
				m_survey->workingSetManager()->addRgbDataset(rgbDataset);
			}
		};
	} else {
		QMessageBox::information(m_parent, "RGB Composite", "No datasets available in survey");
	}
}

void SeismicSurveyRep::AddSeismic(){

	DataUpdatorDialog *dialog = new DataUpdatorDialog("Seismics",m_survey->workingSetManager(),nullptr);
	dialog->show();
}

AbstractGraphicRep::TypeRep SeismicSurveyRep::getTypeGraphicRep() {
	return AbstractGraphicRep::NotDefined;
}

void SeismicSurveyRep::openIsoHorizonInformation(){
	SeismicInformationAggregator* aggregator = new SeismicInformationAggregator(m_survey->workingSetManager());
	ManagerWidget* widget = new ManagerWidget(aggregator);
	widget->show();
}
