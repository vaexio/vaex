<template>
    <v-autocomplete
        v-model="value"
        :items="columns"
        :label="label"
        item-text="name"
        item-value="name"
    >
        <template v-slot:item="{item: column}">
            <v-list-item-avatar>
                <v-icon class="grey white--text" v-if="!column.virtual">view_column</v-icon>
                <v-icon class="grey white--text" v-else>functions</v-icon>
            </v-list-item-avatar>
            <v-list-item-content>
                <v-list-item-title>{{ column.name }}</v-list-item-title>
                <v-list-item-subtitle v-html="column.dtype"></v-list-item-subtitle>
                <v-list-item-subtitle v-if="column.virtual" v-html="column.expression"></v-list-item-subtitle>
            </v-list-item-content>
        </template>
        </template>
    </v-autocomplete>
</template>
