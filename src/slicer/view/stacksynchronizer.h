#ifndef SRC_SLICER_VIEW_STACKSYNCHRONIZER_H
#define SRC_SLICER_VIEW_STACKSYNCHRONIZER_H

#include "stackabledata.h"

#include <QObject>
#include <QDialog>
#include <QMutex>

#include <map>

class QComboBox;
class QListWidget;

class WorkingSetManager;

class StackSynchronizer : public QObject {
	Q_OBJECT
public:

	StackSynchronizer(StackType stackType, QObject* parent = nullptr);
	~StackSynchronizer();

	bool containsData(StackableData* data) const;

	/**
	 * Support QObject destroy signal
	 *
	 * If your data does not support it, you need to remove it yourself with removeData
	 */
	bool addData(StackableData* data);
	bool removeData(StackableData* data);

	void clear();

	StackType stackType() const;
	StackClassType stackClassType() const;

private:
	void synchronizeStack(long index, StackableData* data, AbstractStack* stack);
	std::map<AbstractStack*, long> synchronizeStackInternal(long index, StackableData* data,
			AbstractStack* stack);

	struct StackableDataHolder {
		StackableData* data;
		std::shared_ptr<AbstractStack> stack;
		QMetaObject::Connection syncConn;
		QMetaObject::Connection destroyConn;
	};

	std::vector<StackableDataHolder> m_datas;

	StackType m_stackType;
	StackClassType m_stackClassType;

	QMutex m_mainLock;
	QMutex m_secondaryLock;

	long m_nextIndex = -1;
	StackableData* m_nextData = nullptr;
	AbstractStack* m_nextStack = nullptr;
};

class StackSynchronizerDialog : public QDialog {
	Q_OBJECT
public:
	StackSynchronizerDialog(WorkingSetManager* manager, QWidget* parent=nullptr, Qt::WindowFlags f=Qt::WindowFlags());
	~StackSynchronizerDialog();

	// caller take ownership of synchronizer
	StackSynchronizer* newSynchronizer();

	// caller keep ownership of the synchronizer
	void setSynchronizer(StackSynchronizer* synchronizer);

public slots:
	virtual void accept() override;
	void stackTypeIndexChanged(int index);

private:
	void refreshDataList();
	void initSynchronizer(StackSynchronizer* synchronizer);

	StackSynchronizer* m_synchronizer;
	WorkingSetManager* m_manager;
	StackType m_currentType;

	QComboBox* m_typeComboBox;
	QListWidget* m_dataListWidget;
};

#endif
