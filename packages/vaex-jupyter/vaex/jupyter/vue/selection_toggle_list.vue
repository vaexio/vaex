 <template>
    
    <v-container fluid>
        <h3>{{title}}</h3>
        <v-switch v-model="value" :label="selection_name" :value="selection_name" v-for="selection_name in selection_names" :key="selection_name"></v-switch>
    </v-container>
</template>