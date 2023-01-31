#ifndef ITREEWIDGETITEMDECORATORPROVIDER_H
#define ITREEWIDGETITEMDECORATORPROVIDER_H

class ITreeWidgetItemDecorator;

class ITreeWidgetItemDecoratorProvider {
public:
	virtual ~ITreeWidgetItemDecoratorProvider() {};

	// Decorator ownership is kept by the provider
	// AS 30/062022 : There is no need yet for the decorator to be owned by the provider, this could change in the future
	virtual ITreeWidgetItemDecorator* getTreeWidgetItemDecorator() = 0;
};

#endif // ITREEWIDGETITEMDECORATORPROVIDER_H
