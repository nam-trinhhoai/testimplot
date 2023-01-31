#ifndef FolderDataRep_H
#define FolderDataRep_H

#include <QDialog>
#include <QVBoxLayout>
#include <QListWidget>
#include <QDialogButtonBox>
#include <QDir>
#include <QObject>
#include "abstractgraphicrep.h"
class FolderData;



class widgetAnimation: public QDialog
{
	Q_OBJECT
	public:
	QListWidget* m_listeAnim;

	widgetAnimation(QString path,QWidget *parent= nullptr):QDialog(parent)
	{
		QVBoxLayout* lay = new QVBoxLayout();
		m_listeAnim = new QListWidget();
		QDir dir(path);
		QFileInfoList fileinfo = dir.entryInfoList();

		for (int i = 0; i < fileinfo.size(); ++i)
		{
			 QFileInfo fileInfo = fileinfo.at(i);
			 if(fileInfo.suffix() =="hor" )
			 {
				 QString namepath = fileInfo.baseName();//+"_"+QString::number(m_zScale);
				 m_listeAnim->addItem(namepath);
			 }
		}

		lay->addWidget(m_listeAnim);

		QDialogButtonBox *buttonBox = new QDialogButtonBox(
						QDialogButtonBox::Ok | QDialogButtonBox::Cancel);


		connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
		connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

		lay->addWidget(buttonBox);

		setLayout(lay);
	}
	~widgetAnimation()
	{

	}
};

class FolderDataRep: public AbstractGraphicRep {
Q_OBJECT
public:
	FolderDataRep(FolderData *folderData, AbstractInnerView *parent = 0);
	virtual ~FolderDataRep();

	//AbstractGraphicRep
	virtual bool canBeDisplayed() const override {
		return false;
	}

	virtual QWidget* propertyPanel() override;
	virtual GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	virtual void buildContextMenu(QMenu *menu) override; // MZR 20082021
	virtual TypeRep getTypeGraphicRep() override;

	virtual IData* data() const override;

	void readAnimation(QString path);

private slots:
	void addData();
	void computeReflectivity();
	void addSismageHorizon();
	void computeAttributHorizon();
	void playAnimation();

	void openNVHorizonInformation();
	void openIsoHorizonInformation();
	void loadAnimation();
	void openInformationHorizons();
	void openPicksInformation();
	void openNurbsInformation();
	void openWellInformation();


private:
	FolderData *m_data;
};

#endif
