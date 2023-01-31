#ifndef SRC_INFORMATIONMANAGER_ISOHORIZON_ISOHORIZONINFORMATIONMETADATAWIDGET
#define SRC_INFORMATIONMANAGER_ISOHORIZON_ISOHORIZONINFORMATIONMETADATAWIDGET

#include <QWidget>

class IInformation;

class QPushButton;

class IsoHorizonInformationMetadataWidget : public QWidget {
	Q_OBJECT
public:
	IsoHorizonInformationMetadataWidget(IInformation* information, QWidget* parent=0);
	virtual ~IsoHorizonInformationMetadataWidget();

public slots:
	void openDir();

private:
	IInformation* m_information;

	QPushButton* m_openButton;
	bool m_hasFolder;
	QString m_openDir;
};

#endif // SRC_INFORMATIONMANAGER_NURBS_NURBINFORMATIONMETADATAWIDGET
