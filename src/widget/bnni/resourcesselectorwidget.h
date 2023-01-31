#ifndef SRC_WIDGET_BNNI_RESOURCESELECTORWIDGET_H_
#define SRC_WIDGET_BNNI_RESOURCESELECTORWIDGET_H_

#include "computerresources.h"

#include <QWidget>

#include <map>

class QCheckBox;
class QPushButton;
class QLabel;
class QGridLayout;
class QScrollArea;

class ResourcesSelectorWidget : public QWidget {
	Q_OBJECT
public:
	typedef struct ResourceUI {
		QCheckBox* useCheckBox = nullptr;
		QLabel* hostNameLabel = nullptr;
	} ResourceUI;

	ResourcesSelectorWidget(QWidget* parent=0, Qt::WindowFlags f = Qt::WindowFlags());
	~ResourcesSelectorWidget();

	ComputerResources& resource();
	std::vector<ComputerResource*> getSelectedResources() const;

public slots:
	void addNewResource();
	void addCurrentResource();

private slots:
	void resourceAdded(ComputerResource* resource);
	void resourceRemoved(ComputerResource* resource);

private:
	long m_nextLine = 0;

	QGridLayout* m_resourcesLayout;
	QWidget* m_resourceHolder;
	QScrollArea* m_scrollArea;

	std::map<long, ResourceUI> m_line2ItemMap;
	std::map<long, ComputerResource*> m_line2ResourceMap;
	std::map<long, bool> m_line2UseMap;
	ComputerResources m_resources;
};

#endif
