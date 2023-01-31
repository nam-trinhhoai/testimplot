#ifndef HORIZONANIMMETADATAWIDGET
#define HORIZONANIMMETADATAWIDGET

#include <QWidget>

class IInformation;

class QPushButton;

class HorizonAnimMetadataWidget : public QWidget {
	Q_OBJECT
public:
	HorizonAnimMetadataWidget(IInformation* information, QWidget* parent=0);
	virtual ~HorizonAnimMetadataWidget();

public slots:
	void openDir();

private:
	IInformation* m_information;

	QPushButton* m_openButton;
	bool m_hasFolder;
	QString m_openDir;
};

#endif // HORIZONANIMMETADATAWIDGET
