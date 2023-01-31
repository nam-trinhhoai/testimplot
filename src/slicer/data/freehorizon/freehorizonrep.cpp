
#include <QAction>
#include <QFileInfo>
#include <QMessageBox>

#include <fileInformationWidget.h>
#include <GeotimeProjectManagerWidget.h>
#include "globalconfig.h"
#include <workingsetmanager.h>
#include <horizonAttributComputeDialog.h>
#include <freeHorizonManager.h>
#include <freeHorizonQManager.h>
#include "freehorizonrep.h"
#include "freehorizon.h"

FreeHorizonRep::FreeHorizonRep(FreeHorizon *freehorizon, AbstractInnerView *parent) :
		AbstractGraphicRep(parent) {
	m_data = freehorizon;
	m_name = freehorizon->name();
}

FreeHorizonRep::~FreeHorizonRep() {

}

QWidget* FreeHorizonRep::propertyPanel() {
	return nullptr;
}

GraphicLayer * FreeHorizonRep::layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent) {
	return nullptr;
}

IData* FreeHorizonRep::data() const {
	return m_data;
}

AbstractGraphicRep::TypeRep FreeHorizonRep::getTypeGraphicRep() {
    return AbstractGraphicRep::NotDefined;
}


void FreeHorizonRep::buildContextMenu(QMenu *menu) {
	QAction *attribut = new QAction(tr("Compute attribut"), this);
	menu->addAction(attribut);
	QAction *delete_ = new QAction(tr("unselect"), this);
	menu->addAction(delete_);
	QAction *info = new QAction(tr("info"), this);
	menu->addAction(info);
	QAction *folder = new QAction(tr("folder"), this);
	menu->addAction(folder);
	QAction *exportBtn = new QAction(tr("Export to Sismage"), this);
	menu->addAction(exportBtn);

	connect(attribut, SIGNAL(triggered()), this, SLOT(computeAttribut()));
	connect(delete_, SIGNAL(triggered()), this, SLOT(deleteHorizon()));
	connect(info, SIGNAL(triggered()), this, SLOT(infoHorizon()));
	connect(folder, SIGNAL(triggered()), this, SLOT(folderHorizon()));
	connect(exportBtn, SIGNAL(triggered()), this, SLOT(exportToSismage()));
}

void FreeHorizonRep::computeAttribut()
{
	WorkingSetManager *manager = m_data->workingSetManager();
	if ( manager == nullptr ) { fprintf(stderr, "%s manager null\n"); return; }
	GeotimeProjectManagerWidget* projectManager = manager->getManagerWidget();
	if ( projectManager == nullptr ) { fprintf(stderr, "%s projectManager null\n"); return; }

	HorizonAttributComputeDialog *dialog = new HorizonAttributComputeDialog(nullptr);
	dialog->setProjectManager(projectManager);
	dialog->setWorkingSetManager(m_data->workingSetManager());
	dialog->setInputHorizons(m_data->path(), m_data->name());

	dialog->setVisible(true);
	dialog->show();
}

void FreeHorizonRep::deleteHorizon()
{
	// m_parent->hide();
	// emit deletedRep(this);
	WorkingSetManager *manager = m_data->workingSetManager();
	emit deletedRep(this);
	manager->removeFreeHorizons(m_data);
	this->deleteLater();
}

void FreeHorizonRep::infoHorizon()
{
	QString path = m_data->path();
	QString name = m_data->name();
	std::vector<QString> list0 = FreeHorizonQManager::getAttributData(path);
	std::vector<QString> path0 = FreeHorizonQManager::getAttributPath(path);
	QString txt = "";

	int dimy = 0;
	int dimz = 0;
	FreeHorizonManager::getHorizonDims(path.toStdString(), &dimy, &dimz);

	int nb = 0;
	for (int i=0; i<list0.size(); i++)
	{
		QFileInfo fileInfo(path0[i]);
		QString ext =fileInfo.completeSuffix();
		if ( ext == "raw" || ext == "iso" ) nb++;
	}
	txt += "there are " + QString::number(nb)  + " attributes for the horizon " + name + "\n";
	txt += "dimensions: \ndimy: " + QString::number(dimy) + "\ndimz: " + QString::number(dimz) + "\n";
	for (int i=0; i<list0.size(); i++)
	{
		QFileInfo fileInfo(path0[i]);
		QString ext =fileInfo.completeSuffix();
		if ( ext == "raw" || ext == "iso" )
		{
			QString filesize = FileInformationWidget::getFormatedFileSize(path0[i]);
			if ( ext == "iso" )
			{
				txt += "isochrone ( " + filesize + " )\n";
			}
			else
			{
				QString attribut = 	QString::fromStdString(FreeHorizonManager::typeFromAttributName(list0[i].toStdString()));
				if ( attribut == "spectrum" )
				{
					int nbFreq = FreeHorizonManager::getNbreSpectrumFreq(path0[i].toStdString());
					txt += "spectrum: " + list0[i] + " [ " + QString::number(nbFreq) + " frequencies ] ( " + filesize + " )\n";
				}
				else if ( attribut == "gcc" )
				{
					int nbScales = FreeHorizonManager::getNbreGccScales(path0[i].toStdString());
					txt += "gcc: " + list0[i] + " [ " + QString::number(nbScales) + " scales ] ( " + filesize + " )\n";
				}
				else if ( attribut == "mean" )
				{
					txt += "mean: " + list0[i] + " ( " + filesize + " )\n";
				}
			}
		}
	}
	if ( !txt.isEmpty() )
	{
		QMessageBox messageBox;
		messageBox.information(m_parent, "Info", txt);
	}
//
//	QFile file(filename);
//			if ( file.open(QIODevice::ReadOnly) )
//			{
//			    size = file.size();
//			    file.close();
//			}


}

void FreeHorizonRep::folderHorizon()
{
	GlobalConfig& config = GlobalConfig::getConfig();
	QString cmd = config.fileExplorerProgram() + " " + m_data->path();
	cmd.replace("(", "\\(");
	cmd.replace(")", "\\)");
	system(cmd.toStdString().c_str());


}

void FreeHorizonRep::exportToSismage()
{
	m_data->openSismageExporter();
}

