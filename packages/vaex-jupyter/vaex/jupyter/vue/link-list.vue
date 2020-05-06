 <v-list dense class="pt-0">
    <v-list-item v-for="item in items" :key="item.title" :href="item.url"
                    target="_blank">
        <v-list-item-action>
            <v-icon v-if="item.icon">{{ item.icon }}</v-icon>
            <img v-if="item.img" :src="item.img" style="height: 24px;"></img>
        </v-list-item-action>

        <v-list-item-content>
            <v-list-item-title>{{ item.title }}</v-list-item-title>
        </v-list-item-content>
    </v-list-item>
</v-list>